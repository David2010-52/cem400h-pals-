# pals_basic.py
# 目的：生成“正确、自洽”的3γ事件，然后用三角定位把顶点V估出来，打印MAE/P50/P90。
# 只有 numpy 依赖。想复现实验就改最上面的参数，不要动后面。

import numpy as np
import math

# =========================
# 参数（只改这里就行）
# =========================
R_DET_MM = 250.0   # 探测环半径（总直径≈500 mm）
R_SRC_MM = 150.0   # 顶点采样半径（样品区≈300 mm）
N_EVENTS = 1000    # 一次生成/重建多少个事件
SEED = 42          # 固定整数=可复现；None=每次都不同
SAVE_CSV = True    # 是否把每条结果写到 tri_results.csv

# =========================
# 一些特别简单的小函数
# =========================
def vec_len(v):
    """返回向量长度（浮点数）"""
    return float(np.linalg.norm(v))

def unit(v):
    """返回单位向量；若长度为0就报错防止除0"""
    n = vec_len(v)
    if n == 0.0:
        raise ValueError("zero vector")
    return v / n

# =========================
# 构造等边三角形几何 R（在圆上取0/120/240度）
# =========================
def build_equilateral_R(r_det_mm):
    ang = np.deg2rad([0.0, 120.0, 240.0])  # 弧度
    x = r_det_mm * np.cos(ang)
    y = r_det_mm * np.sin(ang)
    z = np.zeros(3)
    R = np.stack([x, y, z], axis=1)  # 形状 (3,3)
    return R

# =========================
# 在半径 R_SRC_MM 的圆盘里均匀采样 N 个顶点 V（z=0）
# 均匀圆盘要用 r = Rmax * sqrt(U)
# =========================
def sample_vertices_disk(r_src_mm, n, seed):
    if seed is not None:
        np.random.seed(int(seed))
    theta = np.random.uniform(0.0, 2.0*np.pi, size=n)
    rad = r_src_mm * np.sqrt(np.random.rand(n))
    V = np.zeros((n,3), dtype=float)
    V[:,0] = rad * np.cos(theta)
    V[:,1] = rad * np.sin(theta)
    V[:,2] = 0.0
    return V

# =========================
# 事件平面基：给R1,R2,R3，构造ex,ey,n
# =========================
def plane_basis(R1, R2, R3):
    ex = unit(R2 - R1)
    n  = unit(np.cross(R2 - R1, R3 - R1))
    ey = unit(np.cross(n, ex))
    return ex, ey, n

def to_2d(P, R1, ex, ey):
    """把三维点P投到事件平面2D坐标，以R1为原点"""
    v = P - R1
    x = float(np.dot(v, ex))
    y = float(np.dot(v, ey))
    return np.array([x, y], dtype=float)

def from_2d(xy, R1, ex, ey):
    """从事件平面2D坐标还原到三维"""
    return R1 + xy[0]*ex + xy[1]*ey

# =========================
# 生成器：给定 (V,R) 解出自洽能量 E
# A * E = b，其中：
#   [u1x u2x u3x] [E1]   [0]
#   [u1y u2y u3y] [E2] = [0]
#   [  1   1   1 ] [E3]  [1022]
# =========================
def solve_energies_from_VR(V, R):
    ex, ey, n = plane_basis(R[0], R[1], R[2])
    # 3条单位方向（V -> Ri）
    U2 = []  # 存2D分量 (ux, uy)
    for i in range(3):
        ui3 = unit(R[i] - V)
        ux = float(np.dot(ui3, ex))
        uy = float(np.dot(ui3, ey))
        U2.append([ux, uy])
    u1x, u1y = U2[0]
    u2x, u2y = U2[1]
    u3x, u3y = U2[2]

    A = np.array([[u1x, u2x, u3x],
                  [u1y, u2y, u3y],
                  [1.0,  1.0,  1.0]], dtype=float)
    b = np.array([0.0, 0.0, 1022.0], dtype=float)

    # 用最小二乘（更稳）
    E, *_ = np.linalg.lstsq(A, b, rcond=None)

    # 物理性检查：不允许明显负能量
    if np.any(E < -1e-6):
        raise ValueError("Non-physical energies: " + str(E))
    # 小负数（浮点噪声）夹到0
    E = np.maximum(E, 0.0)
    return E  # (3,)

# =========================
# 能量 -> 顶点三角的三顶角（theta12, theta13, theta23）
# cosθij = (Ek^2 - Ei^2 - Ej^2) / (2 Ei Ej)
# =========================
def energies_to_thetas(E):
    E1 = float(E[0])
    E2 = float(E[1])
    E3 = float(E[2])

    def clamp_cos(val):
        if val > 1.0:  val = 1.0
        if val < -1.0: val = -1.0
        return val

    # θ12 对应第三条边是E3
    c12 = (E3*E3 - E1*E1 - E2*E2) / (2.0*E1*E2)
    c13 = (E2*E2 - E1*E1 - E3*E3) / (2.0*E1*E3)
    c23 = (E1*E1 - E2*E2 - E3*E3) / (2.0*E2*E3)

    c12 = clamp_cos(c12)
    c13 = clamp_cos(c13)
    c23 = clamp_cos(c23)

    th12 = math.acos(c12)
    th13 = math.acos(c13)
    th23 = math.acos(c23)
    return th12, th13, th23

# =========================
# 动量闭合残差：|| sum_i E_i * (Ri - V) / ||Ri - V|| ||
# 用来在镜像两个解里挑更合理的那个（越小越好）
# =========================
def closure_error(V3d, R, E):
    s = np.zeros(3, dtype=float)
    for i in range(3):
        ray = R[i] - V3d
        nr = vec_len(ray)
        if nr == 0.0:
            return float("inf")
        s = s + float(E[i]) * (ray / nr)
    return vec_len(s)

# =========================
# 三角定位（Basic版，只有镜像±二选一；不尝试E↔R的3!映射）
# 步骤：
# 1) 降到事件平面2D：r1=(0,0), r2=(L12,0), r3=(x3,y3)
# 2) 用两条顶角 + 两条底边，解出V的二维坐标(x, ±y)
# 3) 用动量闭合在两个镜像中选择更小的
# =========================
def reconstruct_vertex(R, E):
    ex, ey, n = plane_basis(R[0], R[1], R[2])

    # 2D坐标
    r1 = np.array([0.0, 0.0], dtype=float)
    R12 = R[1] - R[0]
    R13 = R[2] - R[0]
    L12 = vec_len(R12)
    r2 = np.array([L12, 0.0], dtype=float)
    r3 = to_2d(R[2], R[0], ex, ey)

    # 两条底边长度
    L13 = vec_len(r3 - r1)

    # 有向夹角phi = angle(r2-r1, r3-r1)
    a = r2 - r1
    b = r3 - r1
    det = a[0]*b[1] - a[1]*b[0]
    dot = a[0]*b[0] + a[1]*b[1]
    phi = math.atan2(det, dot)

    # 能量 -> 三个顶角
    th12, th13, th23 = energies_to_thetas(E)
    s12 = math.sin(th12)
    s13 = math.sin(th13)
    if abs(s12) < 1e-10 or abs(s13) < 1e-10:
        raise RuntimeError("Degenerate angle geometry")

    # 由正弦定理得到比例
    A = L12 / s12
    B = L13 / s13

    # 求alpha（V相对R1R2的极角）
    y_num = A*math.sin(th12) - B*math.sin(th13 + phi)
    x_den = A*math.cos(th12) - B*math.cos(th13 + phi)
    alpha = math.atan2(y_num, x_den)

    # 计算到R1、R2的边长（几何推导后的简式）
    x1 = abs(A * math.sin(th12 + alpha))
    x2 = abs(L12 * math.sin(alpha) / s12)

    # 用余弦定理解出二维坐标 (x, ±y)
    xx = (L12*L12 + x1*x1 - x2*x2) / (2.0*L12)
    yy_sq = x1*x1 - xx*xx
    if yy_sq < 0.0:
        yy_sq = 0.0
    yy = math.sqrt(yy_sq)

    # 镜像两个解
    best_V = None
    best_eps = None
    for sgn in (+1.0, -1.0):
        V2 = np.array([xx, sgn*yy], dtype=float)
        V3 = from_2d(V2, R[0], ex, ey)
        eps = closure_error(V3, R, E)
        if (best_eps is None) or (eps < best_eps):
            best_eps = eps
            best_V = V3

    # 返回三维V和闭合残差
    return best_V, best_eps

# =========================
# 主流程：造正确数据 -> 重建 -> 打印指标 -> 写CSV(可选)
# =========================
def main():
    # 1) 构造示例几何（等边三角形）
    R = build_equilateral_R(R_DET_MM)

    # 2) 采样N个真值V（无噪声）
    V_true_all = sample_vertices_disk(R_SRC_MM, N_EVENTS, SEED)

    V_hat_rows = []
    errs = []
    rows_for_csv = []

    # 3) 对每个V，先解E，再重建回V_hat
    for i in range(len(V_true_all)):
        V = V_true_all[i]
        try:
            E = solve_energies_from_VR(V, R)
        except Exception:
            # 极少数不物理解（或数值问题），直接跳过
            continue

        V_hat, eps = reconstruct_vertex(R, E)
        err = vec_len(V_hat - V)

        V_hat_rows.append(V_hat)
        errs.append(err)

        if SAVE_CSV:
            rows_for_csv.append([V[0], V[1], V[2],
                                 V_hat[0], V_hat[1], V_hat[2],
                                 err, E[0], E[1], E[2]])

    V_hat_rows = np.array(V_hat_rows, dtype=float)
    errs = np.array(errs, dtype=float)

    # 4) 打印指标
    if len(errs) == 0:
        print("No valid events. Try different geometry or parameters.")
        return

    mae = float(np.mean(errs))
    p50 = float(np.percentile(errs, 50))
    p90 = float(np.percentile(errs, 90))

    print("N_eff:", len(errs), "/", len(V_true_all))
    print("MAE (mm):", mae, "P50:", p50, "P90:", p90)

    # 5) 写CSV（可选）
    if SAVE_CSV and len(rows_for_csv) > 0:
        arr = np.array(rows_for_csv, dtype=float)
        header = "Vx_true,Vy_true,Vz_true,Vx_hat,Vy_hat,Vz_hat,err_mm,E1,E2,E3"
        np.savetxt("tri_results.csv", arr, delimiter=",", header=header, comments="")
        print("Saved tri_results.csv")

if __name__ == "__main__":
    main()
