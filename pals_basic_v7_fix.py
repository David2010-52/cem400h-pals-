# pals_basic_v7_fix.py
# 3γ triangulation – angle-only 2D LSQ (+ tiny 3D polish), numpy only
# 关键修正：默认不再做 3! 的 E↔R 穷举（TRY_ALL_MAPPINGS=False）
# 若你的数据确实未知对应关系，再把 TRY_ALL_MAPPINGS=True 打开。

import numpy as np
import math
from itertools import permutations

# ===== 参数（按需改） ==========================================================
R_DET_MM       = 250.0   # 探测器半径（~500 mm 直径等边三点）
R_SRC_MM       = 150.0   # 顶点采样半径（生成器用）
N_EVENTS       = 1000
SEED           = 42      # 固定=可复现；None=每次不同
SAVE_CSV       = True
PRINT_DIAG     = True    # 只对前3条事件打印自检
POLISH_STEPS   = 6       # 3D 抛光步数（0=不抛光；建议 0~6）
W_ANGLE        = 1.0     # 角度项权重
W_CLOSURE      = 0.2     # 闭合项在抛光中的权重（别太大）
TRY_ALL_MAPPINGS = False # ★ 核心开关：已知 E[i]↔R[i] 时设 False

# 2D 多起点网格（以几何质心为中心）
GRID_STEP_MM   = 30.0
GRID_RADIUS_MM = 180.0

# 数值雅可比步长 & 岭
FD_EPS_2D      = 1e-3
FD_EPS_3D      = 1e-3
GN_LAMBDA      = 1e-6

# =============================================================================

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
    return np.stack([x, y, z], axis=1)  # (3,3) 行=探测点，列=x,y,z

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

# ===== 生成器：精确线性解 + 条件数检查 + 非正能量剔除 =========================
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
    s = np.linalg.svd(A, compute_uv=False)
    cond = (s.max()/s.min()) if s.min()>0 else 1e16
    if cond > 1e10:
        raise ValueError("ill-conditioned A, cond=%.2e"%cond)
    E = np.linalg.solve(A, b)
    if np.any(E <= 0.0):
        raise ValueError("non-physical energies")
    return E

# ===== 能量→角度/余弦 =========================================================
def energies_to_thetas_and_cos(E):
    E1,E2,E3 = map(float, E)
    def clamp(x): return 1.0 if x>1 else (-1.0 if x<-1 else x)
    c12 = clamp((E3*E3 - E1*E1 - E2*E2) / (2.0*E1*E2))
    c13 = clamp((E2*E2 - E1*E1 - E3*E3) / (2.0*E1*E3))
    c23 = clamp((E1*E1 - E2*E2 - E3*E3) / (2.0*E2*E3))
    th12, th13, th23 = math.acos(c12), math.acos(c13), math.acos(c23)
    return (th12, th13, th23), (c12, c13, c23)

# ===== 2D 角度余弦最小二乘 =====================================================
def cos_triplet_from_V2(V2, r1, r2, r3):
    u1 = r1 - V2; u1 /= (np.linalg.norm(u1) + 1e-12)
    u2 = r2 - V2; u2 /= (np.linalg.norm(u2) + 1e-12)
    u3 = r3 - V2; u3 /= (np.linalg.norm(u3) + 1e-12)
    c12 = float(np.dot(u1,u2))
    c13 = float(np.dot(u1,u3))
    c23 = float(np.dot(u2,u3))
    return c12, c13, c23

def residual_angle_2d(V2, r1, r2, r3, cos_hat):
    c12, c13, c23 = cos_triplet_from_V2(V2, r1, r2, r3)
    return np.array([c12 - cos_hat[0],
                     c13 - cos_hat[1],
                     c23 - cos_hat[2]], float)

def numeric_jacobian_2d(V2, r1, r2, r3, cos_hat, eps=FD_EPS_2D):
    base = residual_angle_2d(V2, r1, r2, r3, cos_hat)
    J = np.zeros((3,2), float)
    for k in range(2):
        e = np.zeros(2, float); e[k] = eps
        rp = residual_angle_2d(V2 + e, r1, r2, r3, cos_hat)
        rm = residual_angle_2d(V2 - e, r1, r2, r3, cos_hat)
        J[:,k] = (rp - rm) / (2.0*eps)
    return J, base

def solve_V2_from_angles(r1, r2, r3, cos_hat, grid_step=GRID_STEP_MM, grid_rad=GRID_RADIUS_MM):
    C2 = (r1 + r2 + r3) / 3.0
    xs = np.arange(C2[0] - grid_rad, C2[0] + grid_rad + 1e-9, grid_step)
    ys = np.arange(C2[1] - grid_rad, C2[1] + grid_rad + 1e-9, grid_step)
    best_V2, best_cost = None, None
    for x in xs:
        for y in ys:
            V2 = np.array([x, y], float)
            V = V2.copy()
            for _ in range(12):  # 2D GN 步数不需太多
                J, r = numeric_jacobian_2d(V, r1, r2, r3, cos_hat)
                H = J.T @ J + GN_LAMBDA*np.eye(2)
                g = J.T @ r
                try:
                    d = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break
                V_new = V + d
                r_new = residual_angle_2d(V_new, r1, r2, r3, cos_hat)
                if np.dot(r_new, r_new) < np.dot(r, r):
                    V = V_new
                else:
                    break
            cost = float(np.dot(residual_angle_2d(V, r1, r2, r3, cos_hat),
                                residual_angle_2d(V, r1, r2, r3, cos_hat)))
            if (best_cost is None) or (cost < best_cost):
                best_cost, best_V2 = cost, V
    return best_V2, best_cost

# ===== 3D 抛光（角度 + 小权重闭合） ==========================================
def closure_vec(V3d, R, E):
    s = np.zeros(3, float)
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return np.array([np.inf, np.inf, np.inf], float)
        s += float(E[i]) * (ray / nr)
    return s

def cos_from_V3(V3d, R):
    ui = []
    for i in range(3):
        ray = R[i] - V3d
        nr  = vec_len(ray)
        if nr == 0.0: return None
        ui.append(ray / nr)
    u1,u2,u3 = ui
    return (float(np.dot(u1,u2)), float(np.dot(u1,u3)), float(np.dot(u2,u3)))

def residual_joint_3d(V3d, R, E, cos_hat, w_ang=W_ANGLE, w_clo=W_CLOSURE):
    c = cos_from_V3(V3d, R)
    if c is None: return np.array([np.inf]*6, float)
    S = closure_vec(V3d, R, E)
    r = np.zeros(6, float)
    r[0] = w_ang*(c[0] - cos_hat[0])
    r[1] = w_ang*(c[1] - cos_hat[1])
    r[2] = w_ang*(c[2] - cos_hat[2])
    r[3:] = w_clo*S
    return r

def numeric_jacobian_3d(V3d, R, E, cos_hat, eps=FD_EPS_3D, w_ang=W_ANGLE, w_clo=W_CLOSURE):
    base = residual_joint_3d(V3d, R, E, cos_hat, w_ang, w_clo)
    if not np.isfinite(base).all(): return None, base
    J = np.zeros((6,3), float)
    for k in range(3):
        e = np.zeros(3, float); e[k] = eps
        rp = residual_joint_3d(V3d + e, R, E, cos_hat, w_ang, w_clo)
        rm = residual_joint_3d(V3d - e, R, E, cos_hat, w_ang, w_clo)
        if (not np.isfinite(rp).all()) or (not np.isfinite(rm).all()):
            return None, base
        J[:,k] = (rp - rm) / (2.0*eps)
    return J, base

def polish_3d(V0, R, E, cos_hat, steps=POLISH_STEPS):
    if steps <= 0: return V0
    V = V0.copy()
    for _ in range(steps):
        J, r = numeric_jacobian_3d(V, R, E, cos_hat)
        if J is None: break
        H = J.T @ J + GN_LAMBDA*np.eye(3)
        g = J.T @ r
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        V_new = V + d
        r_new = residual_joint_3d(V_new, R, E, cos_hat)
        if np.dot(r_new, r_new) < np.dot(r, r):
            V = V_new
        else:
            break
    return V

# ===== 单映射：角度2D → 3D抛光 → 评分 ========================================
def reconstruct_one_mapping(R, E):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    r1 = np.array([0.0, 0.0], float)
    L12 = vec_len(R[1]-R[0])
    r2 = np.array([L12, 0.0], float)
    r3 = to_2d(R[2], R[0], ex, ey)
    (_, cos_hat) = energies_to_thetas_and_cos(E)
    V2, _ = solve_V2_from_angles(r1, r2, r3, cos_hat,
                                 grid_step=GRID_STEP_MM, grid_rad=GRID_RADIUS_MM)
    V3 = from_2d(V2, R[0], ex, ey)
    V3 = polish_3d(V3, R, E, cos_hat, steps=POLISH_STEPS)
    # 最终评分：角度主、闭合辅
    r_ang = residual_angle_2d(to_2d(V3, R[0], ex, ey), r1, r2, r3, cos_hat)
    S     = closure_vec(V3, R, E)
    score = W_ANGLE*np.dot(r_ang, r_ang) + W_CLOSURE*np.dot(S, S)
    return V3, float(score)

# ===== 映射控制（已知映射就不做 3!） =========================================
def reconstruct_try_mappings(R, E):
    perms = [(0,1,2)] if not TRY_ALL_MAPPINGS else list(permutations([0,1,2]))
    best_V, best_score, best_p = None, None, None
    for p in perms:
        Em = E[list(p)]
        Vh, sc = reconstruct_one_mapping(R, Em)
        if (best_score is None) or (sc < best_score):
            best_V, best_score, best_p = Vh, sc, p
    return best_V, best_score, best_p

# ===== 主流程 =================================================================
def main():
    R = build_equilateral_R(R_DET_MM)
    V_true_all = sample_vertices_disk(R_SRC_MM, N_EVENTS, SEED)

    errs, rows = [], []
    seen = 0

    for idx, V in enumerate(V_true_all):
        try:
            E = solve_energies_from_VR(V, R)
        except Exception:
            continue

        # 生成器自检（前3条）
        if PRINT_DIAG and seen < 3:
            ui = [(R[i]-V)/np.linalg.norm(R[i]-V) for i in range(3)]
            S  = sum(E[i]*ui[i] for i in range(3))
            _, (c12e,c13e,c23e) = energies_to_thetas_and_cos(E)
            c12v = float(np.dot(ui[0],ui[1]))
            c13v = float(np.dot(ui[0],ui[2]))
            c23v = float(np.dot(ui[1],ui[2]))
            print(f"[GENCHK] |S|={np.linalg.norm(S):.2e}, Δc="
                  f"({c12v-c12e:.2e},{c13v-c13e:.2e},{c23v-c23e:.2e})")

        V_hat, sc, perm = reconstruct_try_mappings(R, E)
        if V_hat is None:
            continue

        err = vec_len(V_hat - V)
        errs.append(err)
        seen += 1

        if SAVE_CSV:
            Em = E[list(perm)]
            rows.append([V[0],V[1],V[2], V_hat[0],V_hat[1],V_hat[2], err, Em[0],Em[1],Em[2]])

        if PRINT_DIAG and seen <= 3:
            print(f"[RECON] idx={idx}, perm={perm}, err={err:.3f} mm, score={sc:.3e}")

    if not errs:
        print("No valid events.")
        return

    errs = np.array(errs, float)
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

if __name__ == "__main__":
    main()
