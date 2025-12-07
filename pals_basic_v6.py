# pals_basic_v6.py
# 3γ 三角定位（仅 numpy）
# 关键：闭式几何改为“视角圆相交”（two-chord angle circles intersect）
# 仍保留： (B) 6种 E↔R 映射；(A) 简单联合残差 Gauss–Newton 微调（可关/步数很少）
# 运行：python pals_basic_v6.py

import numpy as np
import math
from itertools import permutations

# ====== 参数（只改这里） =======================================================
R_DET_MM   = 250.0   # 探测环半径（≈500 mm 直径）
R_SRC_MM   = 150.0   # 顶点采样半径（样品区≈300 mm）
N_EVENTS   = 1000
SEED       = 42      # 固定=可复现；None=每次不同
SAVE_CSV   = True
PRINT_PERM = False

GN_STEPS   = 8       # 微调步数（0=不微调）
GN_LAMBDA  = 1e-6
FD_EPS     = 1e-3
W_ANGLE    = 1.0     # 联合残差里角度项权重

# ====== 小工具 ================================================================
def vec_len(v): return float(np.linalg.norm(v))
def unit(v):
    n = vec_len(v)
    if n == 0.0: raise ValueError("zero vector")
    return v / n

def build_equilateral_R(r_det_mm):
    ang = np.deg2rad([0.0, 120.0, 240.0])
    x = r_det_mm * np.cos(ang)
    y = r_det_mm * np.sin(ang)
    z = np.zeros(3)
    return np.stack([x, y, z], axis=1)  # (3,3)

def sample_vertices_disk(r_src_mm, n, seed):
    if seed is not None: np.random.seed(int(seed))
    theta = np.random.uniform(0.0, 2.0*np.pi, size=n)
    rad   = r_src_mm * np.sqrt(np.random.rand(n))
    V = np.zeros((n,3), float)
    V[:,0] = rad * np.cos(theta)
    V[:,1] = rad * np.sin(theta)
    return V

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

# ====== 生成器：给定 (V,R) 解自洽能量 E ======================================
def solve_energies_from_VR(V, R):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    U2 = []
    for i in range(3):
        ui3 = unit(R[i] - V)  # 3D单位方向
        U2.append([float(np.dot(ui3, ex)), float(np.dot(ui3, ey))])
    u1x,u1y = U2[0]; u2x,u2y = U2[1]; u3x,u3y = U2[2]
    A = np.array([[u1x,u2x,u3x],
                  [u1y,u2y,u3y],
                  [1.0, 1.0, 1.0]], float)
    b = np.array([0.0, 0.0, 1022.0], float)
    E, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.any(E < -1e-6): raise ValueError("Non-physical energies: "+str(E))
    return np.maximum(E, 0.0)

# ====== 能量→角度/余弦 =========================================================
def energies_to_thetas_and_cos(E):
    E1,E2,E3 = map(float, E)
    def clamp(x): return 1.0 if x>1 else (-1.0 if x<-1 else x)
    c12 = clamp((E3*E3 - E1*E1 - E2*E2) / (2.0*E1*E2))
    c13 = clamp((E2*E2 - E1*E1 - E3*E3) / (2.0*E1*E3))
    c23 = clamp((E1*E1 - E2*E2 - E3*E3) / (2.0*E2*E3))
    th12, th13, th23 = math.acos(c12), math.acos(c13), math.acos(c23)
    return (th12, th13, th23), (c12, c13, c23)

# ====== 残差/微调（联合：闭合 + 角度一致） ====================================
def closure_vec(V3d, R, E):
    s = np.zeros(3, float)
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return np.array([np.inf, np.inf, np.inf], float)
        s += float(E[i]) * (ray / nr)
    return s

def cos_from_V(V3d, R):
    ui = []
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return None
        ui.append(ray / nr)
    u1,u2,u3 = ui
    return (float(np.dot(u1,u2)), float(np.dot(u1,u3)), float(np.dot(u2,u3)))

def residual_vec(V3d, R, E, cos_hat, w_angle=W_ANGLE):
    S = closure_vec(V3d, R, E)
    if not np.isfinite(S).all(): return np.array([np.inf]*6, float)
    cV = cos_from_V(V3d, R)
    if cV is None: return np.array([np.inf]*6, float)
    r = np.zeros(6, float)
    r[0:3] = S
    r[3]   = w_angle * (cV[0] - cos_hat[0])
    r[4]   = w_angle * (cV[1] - cos_hat[1])
    r[5]   = w_angle * (cV[2] - cos_hat[2])
    return r

def numeric_jacobian(V3d, R, E, cos_hat, eps=FD_EPS, w_angle=W_ANGLE):
    base = residual_vec(V3d, R, E, cos_hat, w_angle)
    if not np.isfinite(base).all(): return None, base
    J = np.zeros((6,3), float)
    for k in range(3):
        e = np.zeros(3, float); e[k] = eps
        r_plus  = residual_vec(V3d + e, R, E, cos_hat, w_angle)
        r_minus = residual_vec(V3d - e, R, E, cos_hat, w_angle)
        if (not np.isfinite(r_plus).all()) or (not np.isfinite(r_minus).all()):
            return None, base
        J[:,k] = (r_plus - r_minus) / (2.0*eps)
    return J, base

def refine_vertex_joint_GN(V0, R, E, cos_hat, steps=GN_STEPS, lam=GN_LAMBDA, eps=FD_EPS, w_angle=W_ANGLE):
    V = V0.astype(float).copy()
    J, r = numeric_jacobian(V, R, E, cos_hat, eps, w_angle)
    if J is None: return V0, float("inf")
    best_V = V.copy()
    best_cost = float(np.dot(r, r))
    for _ in range(steps):
        J, r = numeric_jacobian(V, R, E, cos_hat, eps, w_angle)
        if J is None: break
        H = J.T @ J + lam*np.eye(3)
        g = J.T @ r
        try:
            dV = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        V_new = V + dV
        r_new = residual_vec(V_new, R, E, cos_hat, w_angle)
        if not np.isfinite(r_new).all(): break
        cost_new = float(np.dot(r_new, r_new))
        if cost_new < best_cost:
            best_cost = cost_new
            best_V = V_new
            V = V_new
        else:
            break
    return best_V, best_cost

# ====== 视角圆（2D）：由弦ij与θij得到圆心±与半径 =================================
def _safe_sin(x):
    s = math.sin(x)
    if abs(s) < 1e-9: s = 1e-9 if s >= 0 else -1e-9
    return s

def circle_from_chord_angle(Pi, Pj, theta, sign):
    """
    输入：Pi, Pj (2D)，theta (弧度, 0<theta<pi)，sign=+1/-1（圆心在垂直平分线的哪一侧）
    输出：圆心C, 半径rho
    """
    mid = 0.5*(Pi + Pj)
    v   = Pj - Pi
    L   = vec_len(v)
    if L == 0.0: return None, None
    t = v / L
    n = np.array([-t[1], t[0]])           # 平面内法向（旋转90°）
    s = _safe_sin(theta)
    rho = L / (2.0 * abs(s))              # 半径
    h   = (L / 2.0) / max(math.tan(theta), 1e-9)  # 圆心到弦的距离（可能非常大）
    C   = mid + float(sign) * h * n
    return C, rho

def circle_circle_intersections(C0, r0, C1, r1):
    """返回 0/1/2 个交点（2D）。"""
    d = vec_len(C1 - C0)
    if d > r0 + r1 + 1e-9: return []      # 相离
    if d < abs(r0 - r1) - 1e-9: return [] # 包含
    if d == 0.0 and abs(r0 - r1) < 1e-9:  # 重合（无穷多）——这里不给
        return []
    # 两圆标准交点
    a = (r0*r0 - r1*r1 + d*d) / (2.0*d)
    h_sq = r0*r0 - a*a
    if h_sq < 0.0: h_sq = 0.0
    h = math.sqrt(h_sq)
    p2 = C0 + a * (C1 - C0) / d
    perp = np.array([-(C1 - C0)[1], (C1 - C0)[0]]) / d
    if h == 0.0:
        return [p2]  # 只有一个交点
    return [p2 + h*perp, p2 - h*perp]

# ====== 闭式候选：两条“视角圆”相交（2D），再回到3D ==============================
def candidates_from_two_angle_circles_2d(R, E):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    r1 = np.array([0.0, 0.0], float)
    L12 = vec_len(R[1]-R[0])
    r2 = np.array([L12, 0.0], float)
    r3 = to_2d(R[2], R[0], ex, ey)

    (th12, th13, th23), cos_hat = energies_to_thetas_and_cos(E)

    # 由 R1R2 & θ12 得圆 C12±；由 R1R3 & θ13 得圆 C13±；交点作为候选
    out = []
    for s12 in (+1, -1):
        C12, rho12 = circle_from_chord_angle(r1, r2, th12, s12)
        if C12 is None: continue
        for s13 in (+1, -1):
            C13, rho13 = circle_from_chord_angle(r1, r3, th13, s13)
            if C13 is None: continue
            pts = circle_circle_intersections(C12, rho12, C13, rho13)
            for V2 in pts:
                V3 = from_2d(V2, R[0], ex, ey)
                out.append((V2, V3))
    return out, cos_hat

# ====== 单映射重建：先用“视角圆”给候选，再选最小联合残差；可做少量GN ============
def reconstruct_vertex_one_mapping(R, E):
    cand, cos_hat = candidates_from_two_angle_circles_2d(R, E)
    if not cand:
        # 极端退化：用质心兜底
        V0 = R.mean(axis=0)
        Vh, cost = refine_vertex_joint_GN(V0, R, E, cos_hat, steps=max(GN_STEPS, 6),
                                          lam=GN_LAMBDA, eps=FD_EPS, w_angle=W_ANGLE)
        return Vh, cost

    # 先选联合残差最小的候选
    best_V = None
    best_cost = None
    for _, V3 in cand:
        r = residual_vec(V3, R, E, cos_hat, W_ANGLE)
        if not np.isfinite(r).all(): continue
        c = float(np.dot(r, r))
        if (best_cost is None) or (c < best_cost):
            best_cost = c
            best_V = V3

    # 可选：做几步 GN 抛光
    if GN_STEPS > 0 and best_V is not None:
        Vh, cost = refine_vertex_joint_GN(best_V, R, E, cos_hat, steps=GN_STEPS,
                                          lam=GN_LAMBDA, eps=FD_EPS, w_angle=W_ANGLE)
        if cost < best_cost:
            best_V, best_cost = Vh, cost

    return best_V, best_cost

# ====== 6种映射（取更小） ======================================================
def reconstruct_vertex_try6(R, E):
    best_V, best_cost, best_perm = None, None, None
    for p in permutations([0,1,2]):
        Em = E[list(p)]
        Vh, cost = reconstruct_vertex_one_mapping(R, Em)
        if Vh is None: continue
        if (best_cost is None) or (cost < best_cost):
            best_V, best_cost, best_perm = Vh, cost, p
    return best_V, best_cost, best_perm

# ====== 主流程 ================================================================
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
        if V_hat is None:
            continue
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
