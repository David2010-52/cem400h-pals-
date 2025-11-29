#verion1 9/17/2025
# this is a draft the function are only testing.
# the input and result part are add in, the equal degree confirmation process added.
import numpy as np

def _target_cos_from_E(E):
    """由能量给出理论夹角的 cos 值 [cos12, cos23, cos31]"""
    E1, E2, E3 = map(float, E)
    # 按“第三个是 k”的顺序写
    c12 = (E3**2 - E1**2 - E2**2) / (2.0 * E1 * E2)
    c23 = (E1**2 - E2**2 - E3**2) / (2.0 * E2 * E3)
    c31 = (E2**2 - E3**2 - E1**2) / (2.0 * E3 * E1)
    # 能量有噪声时，数值可能略出 [-1,1]，剪一下避免 nan
    return np.clip(np.array([c12, c23, c31], dtype=float), -1.0, 1.0)

def solve_vertex_from_hits(a, E, max_iter=50, tol_f=1e-10, tol_step=1e-7, damping=1e-9):
    """
    用三条γ的命中三维点 a_i 和能量 E_i，解源点 r=(x,y,z)。
    方程：sum_i E_i * (a_i - r) / ||a_i - r|| = 0
    额外输出：方向两两夹角是否与能量给出的理论夹角一致。
    """
    a = np.asarray(a, dtype=float).reshape(3, 3)
    E = np.asarray(E, dtype=float).reshape(3,)
    I = np.eye(3)

    # 初值：能量加权质心
    r = (E[:, None] * a).sum(axis=0) / (E.sum() + 1e-12)

    converged = False
    condJ = np.inf
    k = 0

    for k in range(max_iter):
        v = a - r
        d = np.linalg.norm(v, axis=1)
        if np.any(d < 1e-12):
            r = r + 1e-6
            v = a - r
            d = np.linalg.norm(v, axis=1)

        u = v / d[:, None]
        F = (E[:, None] * u).sum(axis=0)  # 动量失配

        if np.linalg.norm(F) < tol_f:
            converged = True
            break

        # 雅可比：J = - sum_i (E_i / d_i) * (I - u_i u_i^T)
        J = np.zeros((3, 3))
        for Ei, ui, di in zip(E, u, d):
            J -= (Ei / di) * (I - np.outer(ui, ui))

        # 阻尼
        J_damped = J + damping * I
        try:
            step = np.linalg.solve(J_damped, -F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J_damped, -F, rcond=None)[0]

        r_new = r + step
        if np.linalg.norm(step) < tol_step:
            r = r_new
            converged = True
            break
        r = r_new

        try:
            condJ = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            condJ = np.inf

    # 结束计算各项指标
    v = a - r
    d = np.linalg.norm(v, axis=1)
    u = v / d[:, None]
    F = (E[:, None] * u).sum(axis=0)
    mis = np.linalg.norm(F)

    # 角度一致性检查
    cos_meas = np.array([np.dot(u[0], u[1]), np.dot(u[1], u[2]), np.dot(u[2], u[0])])
    cos_theory = _target_cos_from_E(E)
    angle_err = cos_meas - cos_theory        # 三个 cos 差值
    angle_rmse = float(np.sqrt(np.mean(angle_err**2)))

    # 共面性检查（理论上三方向共面）
    coplanarity = abs(np.dot(np.cross(u[0], u[1]), u[2]))  # 越接近0越好

    info = {
        "misclosure": F,                        # 动量失配向量
        "misclosure_norm": float(mis),          # 其范数（越小越好）
        "condJ": float(condJ),                  # 条件数（大=几何退化）
        "iters": k + 1,
        "converged": bool(converged),
        "cos_measured": cos_meas,               # 实测夹角的 cos
        "cos_theory": cos_theory,               # 能量推导的 cos
        "angle_cos_rmse": angle_rmse,           # 角度一致性RMSE
        "coplanarity_metric": float(coplanarity)# 共面性（≈0理想）
    }
    return r, info

# 便捷外壳
def f(x1, x2, x3, e1, e2, e3):
    a = np.vstack([x1, x2, x3])
    E = np.array([e1, e2, e3], dtype=float)
    return solve_vertex_from_hits(a, E)

if __name__ == "__main__":
    # 用你的真实数据替换这里的示例数
    x1 = np.array([100.0,   0.0, 20.0])  # [x,y,z]
    x2 = np.array([  0.0, 120.0, 25.0])
    x3 = np.array([ -90.0,-110.0, 15.0])
    e1, e2, e3 = 340.0, 360.0, 322.0     # 单道 < 511，总和 ≈ 1022

    r, info = f(x1, x2, x3, e1, e2, e3)
    print("vertex =", r)
    print("misclosure_norm =", info["misclosure_norm"])
    print("angle_cos_rmse =", info.get("angle_cos_rmse"))
    print("coplanarity =", info.get("coplanarity_metric"))