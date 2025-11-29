# pals_basic_v5.py
# 3γ 三角定位（新手风格；仅 numpy）
# 关键改进：Gauss–Newton 目标 = 动量闭合 + 角度一致性（能量→角度）
# 已含：(B) 6种 E↔R 映射尝试；(A) 数值雅可比的 Gauss–Newton 微调（联合残差）
# 不再做大网格搜索（容易被伪解拖慢），改为稳健的联合残差最小化。

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

GN_STEPS   = 20      # GN 最大迭代步数（可以 12~30）
GN_LAMBDA  = 1e-6    # 岭参数（防奇异）
FD_EPS     = 1e-3    # 数值雅可比的有限差分步长（mm）
W_ANGLE    = 1.0     # 角度残差的权重（相对闭合向量的权重；可在 0.5~5 调）

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

# 能量→三顶角（以及 cos 值）
def energies_to_thetas_and_cos(E):
    E1,E2,E3 = map(float, E)
    def clamp(x): return 1.0 if x>1 else (-1.0 if x<-1 else x)
    c12 = clamp((E3*E3 - E1*E1 - E2*E2) / (2.0*E1*E2))
    c13 = clamp((E2*E2 - E1*E1 - E3*E3) / (2.0*E1*E3))
    c23 = clamp((E1*E1 - E2*E2 - E3*E3) / (2.0*E2*E3))
    th12, th13, th23 = math.acos(c12), math.acos(c13), math.acos(c23)
    return (th12, th13, th23), (c12, c13, c23)

# 动量闭合向量 S(V)
def closure_vec(V3d, R, E):
    s = np.zeros(3, float)
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return np.array([np.inf, np.inf, np.inf], float)
        s += float(E[i]) * (ray / nr)   # E_i * u_i
    return s

# 从 V 计算方向余弦（角度）
def cos_from_V(V3d, R):
    ui = []
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return None
        ui.append(ray / nr)
    u1,u2,u3 = ui
    c12 = float(np.dot(u1,u2))
    c13 = float(np.dot(u1,u3))
    c23 = float(np.dot(u2,u3))
    return (c12, c13, c23)

# 联合残差 r(V)：[Sx, Sy, Sz, w*(c12(V)-ĉ12), w*(c13(V)-ĉ13), w*(c23(V)-ĉ23)]
def residual_vec(V3d, R, E, cos_hat, w_angle=W_ANGLE):
    S = closure_vec(V3d, R, E)
    if not np.isfinite(S).all():
        return np.array([np.inf]*6, float)
    cV = cos_from_V(V3d, R)
    if cV is None:
        return np.array([np.inf]*6, float)
    r = np.zeros(6, float)
    r[0:3] = S
    r[3]   = w_angle * (cV[0] - cos_hat[0])
    r[4]   = w_angle * (cV[1] - cos_hat[1])
    r[5]   = w_angle * (cV[2] - cos_hat[2])
    return r

# 数值雅可比（有限差分，中央差分）
def numeric_jacobian(V3d, R, E, cos_hat, eps=FD_EPS, w_angle=W_ANGLE):
    base = residual_vec(V3d, R, E, cos_hat, w_angle)
    if not np.isfinite(base).all():
        return None, base
    J = np.zeros((6,3), float)
    for k in range(3):
        e = np.zeros(3, float); e[k] = eps
        r_plus  = residual_vec(V3d + e, R, E, cos_hat, w_angle)
        r_minus = residual_vec(V3d - e, R, E, cos_hat, w_angle)
        if (not np.isfinite(r_plus).all()) or (not np.isfinite(r_minus).all()):
            return None, base
        J[:,k] = (r_plus - r_minus) / (2.0*eps)
    return J, base

# 朴素的 Gauss–Newton：最小化 ||r(V)||^2
def refine_vertex_joint_GN(V0, R, E, cos_hat, steps=GN_STEPS, lam=GN_LAMBDA, eps=FD_EPS, w_angle=W_ANGLE):
    V = V0.astype(float).copy()
    best_V = V.copy()
    J, r = numeric_jacobian(V, R, E, cos_hat, eps, w_angle)
    if J is None:
        return V0, float("inf")
    best_cost = float(np.dot(r, r))
    for _ in range(steps):
        J, r = numeric_jacobian(V, R, E, cos_hat, eps, w_angle)
        if J is None:
            break
        H = J.T @ J + lam*np.eye(3)
        g = J.T @ r
        try:
            dV = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        V_new = V + dV
        r_new = residual_vec(V_new, R, E, cos_hat, w_angle)
        if not np.isfinite(r_new).all():
            break
        cost_new = float(np.dot(r_new, r_new))
        if cost_new < best_cost:
            best_cost = cost_new
            best_V = V_new
            V = V_new
        else:
            # 步长不降成本就停（也可在这里加一次小步长尝试）
            break
    return best_V, math.sqrt(best_cost)

# 闭式几何（拿个近似初值；如果失败就用几何中心做初值）
def reconstruct_vertex_closed(R, E):
    # 用 R1 为原点在事件平面求一个闭式近似（跟 v4 一样）
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

    (th12, th13, th23), _ = energies_to_thetas_and_cos(E)
    s12, s13 = math.sin(th12), math.sin(th13)
    if abs(s12)<1e-10 or abs(s13)<1e-10:
        return None

    A = L12 / s12; B = L13 / s13
    y_num = A*math.sin(th12) - B*math.sin(th13 + phi)
    x_den = A*math.cos(th12) - B*math.cos(th13 + phi)
    alpha = math.atan2(y_num, x_den)
    x1 = abs(A * math.sin(th12 + alpha))
    x2 = abs(L12 * math.sin(alpha) / s12)
    xx = (L12*L12 + x1*x1 - x2*x2) / (2.0*L12)
    yy_sq = x1*x1 - xx*xx
    yy = math.sqrt(yy_sq) if yy_sq>0 else 0.0

    best_V = None
    best_eps = None
    for sgn in (+1.0, -1.0):
        V2 = np.array([xx, sgn*yy], float)
        V3 = from_2d(V2, R[0], ex, ey)
        eps = vec_len(closure_vec(V3, R, E))
        if (best_eps is None) or (eps < best_eps):
            best_V, best_eps = V3, eps
    return best_V

# 单一映射：初值 = 闭式（若失败则用探测器质心），再做“联合残差”的 GN
def reconstruct_vertex_one_mapping(R, E):
    V0 = reconstruct_vertex_closed(R, E)
    if V0 is None:
        V0 = R.mean(axis=0)  # 保险初值
    _, cos_hat = energies_to_thetas_and_cos(E)
    Vh, cost = refine_vertex_joint_GN(V0, R, E, cos_hat,
                                      steps=GN_STEPS, lam=GN_LAMBDA,
                                      eps=FD_EPS, w_angle=W_ANGLE)
    return Vh, cost

# (B) 6种映射：取联合残差更小者
def reconstruct_vertex_try6(R, E):
    best_V, best_cost, best_perm = None, None, None
    for p in permutations([0,1,2]):
        Em = E[list(p)]
        Vh, cost = reconstruct_vertex_one_mapping(R, Em)
        if (best_cost is None) or (cost < best_cost):
            best_V, best_cost, best_perm = Vh, cost, p
    return best_V, best_cost, best_perm

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

        V_hat, cost, perm = reconstruct_vertex_try6(R, E)
        perm_count[perm] += 1
        Em = E[list(perm)]

        err = vec_len(V_hat - V)
        errs.append(err)
        if SAVE_CSV:
            rows.append([V[0],V[1],V[2], V_hat[0],V_hat[1],V_hat[2], err, Em[0],Em[1],Em[2]])

        if PRINT_PERM:
            print("perm:", perm, "cost:", cost, "err:", err)

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
