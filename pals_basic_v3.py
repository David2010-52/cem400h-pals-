# pals_basic_v3.py
# 3γ 三角定位（新手风格，只有 numpy）
# 已含：
#   (B) 6种 E↔R 映射尝试（取动量闭合更小者）
#   (G) 2D 粗网格搜（事件平面，找更好的初值）
#   (A) Gauss–Newton 微调（3D，对 V 做少量步数最小二乘）
# 运行：python pals_basic_v3.py

import numpy as np
import math
from itertools import permutations

# ===== 参数（只改这里） =====
R_DET_MM   = 250.0   # 探测环半径（≈500 mm 直径）
R_SRC_MM   = 150.0   # 顶点采样半径（样品区≈300 mm）
N_EVENTS   = 1000    # 一次生成/重建的事件数
SEED       = 42      # 固定=可复现；None=每次不同
SAVE_CSV   = True    # 是否导出 tri_results.csv
PRINT_PERM = False   # 是否打印每个事件采用的 E↔R 映射

# 网格搜索参数（越小越准但更慢；建议先用默认）
GRID_RAD_SCALE = 0.80   # 网格半径 = 0.8 × 探测半径
GRID_STEP_MM   = 20.0   # 网格步长（mm）

# ===== 小工具函数 =====
def vec_len(v): return float(np.linalg.norm(v))
def unit(v):
    n = vec_len(v)
    if n == 0.0: raise ValueError("zero vector")
    return v / n

# 等边三角探测几何
def build_equilateral_R(r_det_mm):
    ang = np.deg2rad([0.0, 120.0, 240.0])
    x = r_det_mm * np.cos(ang)
    y = r_det_mm * np.sin(ang)
    z = np.zeros(3)
    return np.stack([x, y, z], axis=1)  # (3,3)

# 圆盘均匀采样顶点（z=0）
def sample_vertices_disk(r_src_mm, n, seed):
    if seed is not None: np.random.seed(int(seed))
    theta = np.random.uniform(0.0, 2.0*np.pi, size=n)
    rad   = r_src_mm * np.sqrt(np.random.rand(n))
    V = np.zeros((n,3), float)
    V[:,0] = rad * np.cos(theta)
    V[:,1] = rad * np.sin(theta)
    return V

# 事件平面基
def plane_basis(R1, R2, R3):
    ex = unit(R2 - R1)
    n  = unit(np.cross(R2 - R1, R3 - R1))
    ey = unit(np.cross(n, ex))
    return ex, ey, n

def to_2d(P, R1, ex, ey):
    v = P - R1
    return np.array([float(np.dot(v, ex)), float(np.dot(v, ey))], float)

def from_2d(xy, R1, ex, ey):
    return R1 + xy[0]*ex + xy[1]*ey

# 生成器：给定 (V,R) 解自洽 E（两分量动量闭合 + 总能量）
def solve_energies_from_VR(V, R):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    U2 = []
    for i in range(3):
        ui3 = unit(R[i] - V)
        U2.append([float(np.dot(ui3, ex)), float(np.dot(ui3, ey))])
    u1x,u1y = U2[0]; u2x,u2y = U2[1]; u3x,u3y = U2[2]
    A = np.array([[u1x,u2x,u3x],
                  [u1y,u2y,u3y],
                  [1.0, 1.0, 1.0]], float)
    b = np.array([0.0, 0.0, 1022.0], float)
    E, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.any(E < -1e-6): raise ValueError("Non-physical energies: "+str(E))
    return np.maximum(E, 0.0)

# 能量→三顶角
def energies_to_thetas(E):
    E1,E2,E3 = map(float, E)
    def clamp(x): return 1.0 if x>1 else (-1.0 if x<-1 else x)
    c12 = clamp((E3*E3 - E1*E1 - E2*E2) / (2.0*E1*E2))
    c13 = clamp((E2*E2 - E1*E1 - E3*E3) / (2.0*E1*E3))
    c23 = clamp((E1*E1 - E2*E2 - E3*E3) / (2.0*E2*E3))
    return math.acos(c12), math.acos(c13), math.acos(c23)

# 动量闭合残差（越小越好）
def closure_error(V3d, R, E):
    s = np.zeros(3, float)
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return float("inf")
        s += float(E[i]) * (ray / nr)
    return vec_len(s)

# 闭式几何（镜像±二选一）
def reconstruct_vertex_closed(R, E):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    r1 = np.array([0.0, 0.0], float)
    L12 = vec_len(R[1]-R[0])
    r2 = np.array([L12, 0.0], float)
    r3 = to_2d(R[2], R[0], ex, ey)

    L13 = vec_len(r3 - r1)
    a = r2 - r1; b = r3 - r1
    det = a[0]*b[1] - a[1]*b[0]
    dot = a[0]*b[0] + a[1]*b[1]
    phi = math.atan2(det, dot)

    th12, th13, th23 = energies_to_thetas(E)
    s12, s13 = math.sin(th12), math.sin(th13)
    if abs(s12)<1e-10 or abs(s13)<1e-10:
        return None, float("inf")
    A = L12 / s12; B = L13 / s13
    y_num = A*math.sin(th12) - B*math.sin(th13 + phi)
    x_den = A*math.cos(th12) - B*math.cos(th13 + phi)
    alpha = math.atan2(y_num, x_den)
    x1 = abs(A * math.sin(th12 + alpha))
    x2 = abs(L12 * math.sin(alpha) / s12)
    xx = (L12*L12 + x1*x1 - x2*x2) / (2.0*L12)
    yy_sq = x1*x1 - xx*xx
    yy = math.sqrt(yy_sq) if yy_sq>0 else 0.0

    best_V, best_eps = None, None
    for sgn in (+1.0, -1.0):
        V2 = np.array([xx, sgn*yy], float)
        V3 = from_2d(V2, R[0], ex, ey)
        eps = closure_error(V3, R, E)
        if (best_eps is None) or (eps < best_eps):
            best_V, best_eps = V3, eps
    return best_V, best_eps

# (G) 2D 粗网格搜：在事件平面内找闭合更小的初值
def reconstruct_vertex_grid(R, E, step_mm=GRID_STEP_MM, rad_scale=GRID_RAD_SCALE):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    # 用探测三点到“几何中心”的最大半径估一个搜索半径
    Rcen = R.mean(axis=0)
    radR = max(vec_len(R[i]-Rcen) for i in range(3))
    rad  = rad_scale * radR

    best_V, best_eps = None, None
    xs = np.arange(-rad, rad+1e-6, step_mm)
    ys = np.arange(-rad, rad+1e-6, step_mm)
    for x in xs:
        for y in ys:
            V2 = np.array([x,y], float)
            V3 = from_2d(V2, R[0], ex, ey)
            eps = closure_error(V3, R, E)
            if (best_eps is None) or (eps < best_eps):
                best_V, best_eps = V3, eps
    return best_V, best_eps

# (A) Gauss–Newton 微调（3D）
def refine_vertex_gauss_newton(V0, R, E, steps=6, lam=1e-6):
    V = V0.astype(float).copy()
    for _ in range(steps):
        S = np.zeros(3, float)
        J = np.zeros((3,3), float)
        for i in range(3):
            ri = R[i] - V
            nr = vec_len(ri)
            if nr == 0.0: return V, float("inf")
            ui = ri / nr
            S += float(E[i]) * ui
            I = np.eye(3)
            dUi_dV = -(I / nr - np.outer(ri, ri) / (nr**3))
            J += float(E[i]) * dUi_dV
        H = J.T @ J + lam*np.eye(3)
        g = J.T @ S
        try:
            dV = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        V_new = V + dV
        if closure_error(V_new, R, E) < closure_error(V, R, E):
            V = V_new
        else:
            break
    return V, closure_error(V, R, E)

# 综合：对给定映射下的 E，先闭式，再网格，取更好 → 微调
def reconstruct_vertex_one_mapping(R, E):
    Vc, ec = reconstruct_vertex_closed(R, E)
    Vg, eg = reconstruct_vertex_grid(R, E)
    # 选更好的初值
    if ec <= eg:
        V0, e0 = Vc, ec
    else:
        V0, e0 = Vg, eg
    # 微调
    Vh, eh = refine_vertex_gauss_newton(V0, R, E, steps=6, lam=1e-6)
    if eh <= e0: return Vh, eh
    return V0, e0

# (B) 6种映射：取闭合更小者
def reconstruct_vertex_try6(R, E):
    best_V, best_eps, best_perm = None, None, None
    for p in permutations([0,1,2]):
        Em = E[list(p)]
        Vh, eps = reconstruct_vertex_one_mapping(R, Em)
        if (best_eps is None) or (eps < best_eps):
            best_V, best_eps, best_perm = Vh, eps, p
    return best_V, best_eps, best_perm

# ===== 主流程 =====
def main():
    R = build_equilateral_R(R_DET_MM)
    V_true_all = sample_vertices_disk(R_SRC_MM, N_EVENTS, SEED)

    errs = []
    rows = []
    perm_count = {p:0 for p in permutations([0,1,2])}

    for V in V_true_all:
        try:
            E = solve_energies_from_VR(V, R)
        except Exception:
            continue

        V_hat0, eps0, perm = reconstruct_vertex_try6(R, E)
        perm_count[perm] += 1
        Em = E[list(perm)]  # 采用该映射的能量顺序

        # 保险：再做一次微调（一般已经很小了）
        V_hat, eps = refine_vertex_gauss_newton(V_hat0, R, Em, steps=4, lam=1e-6)
        if eps > eps0:
            V_hat, eps = V_hat0, eps0

        err = vec_len(V_hat - V)
        errs.append(err)
        if SAVE_CSV:
            rows.append([V[0],V[1],V[2], V_hat[0],V_hat[1],V_hat[2], err, Em[0],Em[1],Em[2]])

        if PRINT_PERM:
            print("perm:", perm, "eps:", eps, "err:", err)

    errs = np.array(errs, float)
    if len(errs)==0:
        print("No valid events.")
        return

    mae = float(np.mean(errs))
    p50 = float(np.percentile(errs, 50))
    p90 = float(np.percentile(errs, 90))

    print("N_eff:", len(errs), "/", len(V_true_all))
    print("MAE (mm):", mae, "P50:", p50, "P90:", p90)

    if SAVE_CSV and rows:
        arr = np.array(rows, float)
        header = "Vx_true,Vy_true,Vz_true,Vx_hat,Vy_hat,Vz_hat,err_mm,E1,E2,E3"
        np.savetxt("tri_results.csv", arr, delimiter=",", header=header, comments="")
        print("Saved tri_results.csv")

    if PRINT_PERM:
        print("perm counts:", perm_count)

if __name__ == "__main__":
    main()
