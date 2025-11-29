# first draft 9/17/2025
# main.py
# Purpose: Library of functions to reconstruct the emission vertex r=(x,y,z)
#          from three gamma hits using energy-weighted momentum closure.
# Notes:
#   - DOI is intentionally ignored (to be added later if needed).
#   - Pairwise angles between directions are energy-determined; we expose QC checks.

from __future__ import annotations
import numpy as np

def _target_cos_from_E(E: np.ndarray) -> np.ndarray:
    """
    Return theoretical cosines of pairwise angles [cos12, cos23, cos31]
    derived from momentum closure sum_i E_i * u_i = 0:
        cos(theta_ij) = (E_k^2 - E_i^2 - E_j^2) / (2 E_i E_j),  k != i != j
    Values are clipped to [-1, 1] for numerical robustness.
    """
    E1, E2, E3 = map(float, E)
    c12 = (E3**2 - E1**2 - E2**2) / (2.0 * E1 * E2)
    c23 = (E1**2 - E2**2 - E3**2) / (2.0 * E2 * E3)
    c31 = (E2**2 - E3**2 - E1**2) / (2.0 * E3 * E1)
    return np.clip(np.array([c12, c23, c31], dtype=float), -1.0, 1.0)

def cyclic_sector_angles_from_u(u):
    """
    Given 3 coplanar unit vectors u[0..2], return the three adjacent sector
    angles around the circle (in degrees), which ALWAYS sum to 360°.
    Order is ascending by azimuth (not tied to (θ12,θ23,θ31) naming).
    """
    import numpy as np
    u = np.asarray(u, float).reshape(3, 3)

    # Plane basis
    n = np.cross(u[0], u[1])
    if np.linalg.norm(n) < 1e-12:
        n = np.cross(u[0], u[2])
    if np.linalg.norm(n) < 1e-12:
        # almost collinear -> define trivial partition
        return np.array([0.0, 0.0, 360.0])

    n = n / np.linalg.norm(n)
    x_axis = u[0] / (np.linalg.norm(u[0]) + 1e-12)
    y_axis = np.cross(n, x_axis); y_axis /= (np.linalg.norm(y_axis) + 1e-12)

    # Azimuths
    ang = []
    for i in range(3):
        x = float(np.dot(u[i], x_axis))
        y = float(np.dot(u[i], y_axis))
        a = np.degrees(np.arctan2(y, x))
        if a < 0: a += 360.0
        ang.append(a)
    ang = np.array(ang)

    # Sort and take cyclic gaps
    idx = np.argsort(ang)
    a_sorted = ang[idx]
    gaps = np.array([
        a_sorted[1] - a_sorted[0],
        a_sorted[2] - a_sorted[1],
        360.0 - (a_sorted[2] - a_sorted[0]),
    ])
    return gaps  # sum(gaps) == 360


def _weiszfeld(a: np.ndarray, w: np.ndarray,
               r0: np.ndarray | None = None,
               tol: float = 1e-8, max_iter: int = 10000) -> tuple[np.ndarray, int]:
    """
    Weighted geometric median (Weiszfeld). Always converges except when r hits a_i。
    Returns (r, iters).  Here a: (3,3), w: (3,).
    """
    a = np.asarray(a, float); w = np.asarray(w, float)
    r = (w[:, None] * a).sum(axis=0) / (w.sum() + 1e-12) if r0 is None else np.asarray(r0, float)
    for k in range(max_iter):
        v = r - a
        d = np.linalg.norm(v, axis=1)
        d = np.where(d < 1e-12, 1e-12, d)
        num = (w[:, None] * a / d[:, None]).sum(axis=0)
        den = (w / d).sum()
        r_new = num / den
        if np.linalg.norm(r_new - r) < tol:
            return r_new, k + 1
        r = r_new
    return r, max_iter

def angles_from_energies(e1: float, e2: float, e3: float, degrees: bool = True):
    """
    Theoretical pairwise angles (θ12, θ23, θ31) determined purely by energies.
    """
    E1, E2, E3 = map(float, (e1, e2, e3))
    c12 = (E3**2 - E1**2 - E2**2) / (2.0 * E1 * E2)
    c23 = (E1**2 - E2**2 - E3**2) / (2.0 * E2 * E3)
    c31 = (E2**2 - E3**2 - E1**2) / (2.0 * E3 * E1)
    cosines = np.clip([c12, c23, c31], -1.0, 1.0)
    ang = np.arccos(cosines)
    return np.degrees(ang) if degrees else ang

def angles_from_geometry(a, r, degrees: bool = True):
    """
    Measured pairwise angles (θ12, θ23, θ31) from geometry (directions a_i - r).
    """
    a = np.asarray(a, float).reshape(3, 3)
    r = np.asarray(r, float).reshape(3,)
    v = a - r
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos12, cos23, cos31 = np.dot(u[0], u[1]), np.dot(u[1], u[2]), np.dot(u[2], u[0])
    ang = np.arccos(np.clip([cos12, cos23, cos31], -1.0, 1.0))
    return np.degrees(ang) if degrees else ang


def _directions_residual_jacobian(r: np.ndarray,
                                  a: np.ndarray,
                                  E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute unit directions u, residual F, Jacobian J, and distances d,
    given vertex r, hit points a (3x3), and energies E (3,).
    """
    I = np.eye(3)
    v = a - r
    d = np.linalg.norm(v, axis=1)
    d = np.where(d < 1e-15, 1e-15, d)   # avoid division by zero
    u = v / d[:, None]
    F = (E[:, None] * u).sum(axis=0)
    J = np.zeros((3, 3))
    for Ei, ui, di in zip(E, u, d):
        J -= (Ei / di) * (I - np.outer(ui, ui))
    return u, F, J, d

def _weiszfeld(a: np.ndarray, w: np.ndarray,
               r0: np.ndarray | None = None,
               tol: float = 1e-8, max_iter: int = 10000) -> tuple[np.ndarray, int]:
    """
    Weighted geometric median (Weiszfeld). Robust fallback when Newton fails.
    Returns (r, iters).  a: (3,3) hit points; w: (3,) weights (use energies).
    """
    a = np.asarray(a, float); w = np.asarray(w, float)
    r = (w[:, None] * a).sum(axis=0) / (w.sum() + 1e-12) if r0 is None else np.asarray(r0, float)
    for k in range(max_iter):
        v = r - a
        d = np.linalg.norm(v, axis=1)
        d = np.where(d < 1e-12, 1e-12, d)  # avoid zero distance
        num = (w[:, None] * a / d[:, None]).sum(axis=0)
        den = (w / d).sum()
        r_new = num / den
        if np.linalg.norm(r_new - r) < tol:
            return r_new, k + 1
        r = r_new
    return r, max_iter

def solve_vertex_from_hits(a: np.ndarray,
                           E: np.ndarray,
                           max_iter: int = 50,
                           tol_f: float = 1e-10,
                           tol_step: float = 1e-7,
                           damping: float = 1e-9) -> tuple[np.ndarray, dict]:
    """
    Solve r from three 3D hit points a_i and energies E_i using:
        sum_i E_i * (a_i - r) / ||a_i - r|| = 0
    Returns (r, info) where info contains QC metrics.
    """
    a = np.asarray(a, dtype=float).reshape(3, 3)
    E = np.asarray(E, dtype=float).reshape(3,)
    I = np.eye(3)

    # Energy-weighted centroid as initialization
    r = (E[:, None] * a).sum(axis=0) / (E.sum() + 1e-12)
    converged = False
    J_damped = None

    for k in range(max_iter):
        u, F, J, d = _directions_residual_jacobian(r, a, E)
        if np.linalg.norm(F) < tol_f:
            converged = True
            break
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
    # ←—— for 循环到这里结束（这一行已经是“循环外”缩进）——→
    r = (E[:, None] * a).sum(axis=0) / (E.sum() + 1e-12)  # 你已有
    r_new = r  # ← 新增：初始化，防止未定义

    r = r_new  # 用最后一次 Newton 的结果

    # === A2 兜底：放在这里（循环外 & Final diagnostics 之前）===
    u_tmp, F_tmp, J_tmp, d_tmp = _directions_residual_jacobian(r, a, E)
    used_fallback = False
    fallback_iters = 0
    if (not converged) or (np.linalg.norm(F_tmp) > 1e-3 * float(E.sum())):
        r_w, it_w = _weiszfeld(a, E, r0=r)
        r = r_w
        used_fallback = True
        fallback_iters = it_w
        converged = True
    # === 兜底结束 ===
    # Final diagnostics
    u, F, J, d = _directions_residual_jacobian(r, a, E)
    cos_meas = np.array([np.dot(u[0], u[1]), np.dot(u[1], u[2]), np.dot(u[2], u[0])])
    cos_theory = _target_cos_from_E(E)
    angle_rmse = float(np.sqrt(np.mean((cos_meas - cos_theory) ** 2)))
    coplanarity = float(abs(np.dot(np.cross(u[0], u[1]), u[2])))
    # ===== 新增：显式角度（度数）及其误差 =====
    angles_measured_deg = np.degrees(np.arccos(np.clip(cos_meas, -1.0, 1.0)))
    angles_theory_deg = np.degrees(np.arccos(np.clip(cos_theory, -1.0, 1.0)))
    angle_deg_errors = angles_measured_deg - angles_theory_deg
    angle_deg_rmse = float(np.sqrt(np.mean(angle_deg_errors ** 2)))
    # 扇区角（必和为 360°）
    angles_measured_cyclic_deg = cyclic_sector_angles_from_u(u)
    # 理论扇区角：能量角本身就对应外角，三者求和≈360，这里直接复制一份便于对比
    angles_theory_cyclic_deg = angles_theory_deg.copy()
    # =======================================
    J_damped_final = J + damping * I
    try:
        condJ = float(np.linalg.cond(J_damped_final))
    except np.linalg.LinAlgError:
        condJ = float("inf")

    # === Unweighted cross-check: ignore energies ===
    r_unw, it_unw = _weiszfeld(a, np.ones(3), r0=r)  # 等权几何中值
    v_unw = a - r_unw
    u_unw = v_unw / np.linalg.norm(v_unw, axis=1, keepdims=True)
    cos_meas_unw = np.array([np.dot(u_unw[0], u_unw[1]),
                             np.dot(u_unw[1], u_unw[2]),
                             np.dot(u_unw[2], u_unw[0])])
    # 与能量推导角对比（度数 RMSE）
    angles_measured_unw_deg = np.degrees(np.arccos(np.clip(cos_meas_unw, -1.0, 1.0)))
    angles_theory_deg = np.degrees(np.arccos(np.clip(cos_theory, -1.0, 1.0)))  # 你已有
    angle_deg_rmse_unw = float(np.sqrt(np.mean((angles_measured_unw_deg - angles_theory_deg) ** 2)))

    # 两种解的距离（稳健性指标）
    delta_r_unw = float(np.linalg.norm(r - r_unw))

    info = {
        "misclosure": F,
        "misclosure_norm": float(np.linalg.norm(F)),
        "condJ": condJ,
        "iters": k + 1,
        "converged": bool(converged),
        "cos_measured": cos_meas,
        "cos_theory": cos_theory,
        "angle_cos_rmse": angle_rmse,
        "coplanarity_metric": coplanarity,
    }
    info.update({
        "angles_measured_deg": angles_measured_deg,
        "angles_theory_deg": angles_theory_deg,
        "angle_deg_errors": angle_deg_errors,
        "angle_deg_rmse": angle_deg_rmse,
        "used_fallback": used_fallback,  # NEW
        "fallback": "weiszfeld" if used_fallback else "",  # NEW
        "fallback_iters": fallback_iters,  # NEW
        "angles_measured_cyclic_deg": angles_measured_cyclic_deg,
        "angles_theory_cyclic_deg": angles_theory_cyclic_deg,
    })
    info.update({
        "r_unweighted": r_unw,
        "delta_r_unweighted": delta_r_unw,
        "angles_unweighted_deg": angles_measured_unw_deg,
        "angle_deg_rmse_unweighted": angle_deg_rmse_unw,
    })
    return r, info

def f(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
      e1: float, e2: float, e3: float) -> tuple[np.ndarray, dict]:
    """
    Convenience wrapper:
        f(x1, x2, x3, e1, e2, e3) -> (r, info)
    x1, x2, x3 are 3D points; e1, e2, e3 are energies.
    """
    a = np.vstack([x1, x2, x3])
    E = np.array([e1, e2, e3], dtype=float)
    return solve_vertex_from_hits(a, E)

def pretty_print_result(r: np.ndarray, info: dict) -> None:
    """Human-readable console output (now includes angle comparison)."""
    print("=== Vertex Reconstruction Result ===")
    print(f"vertex (x, y, z): {r}")
    print(f"converged: {info.get('converged')} | iters: {info.get('iters')}")
    if 'misclosure_norm' in info:
        print(f"misclosure_norm: {info['misclosure_norm']:.6e}")
    if 'angle_cos_rmse' in info:
        print(f"angle_cos_rmse:  {info['angle_cos_rmse']:.6e}")
    if 'coplanarity_metric' in info:
        print(f"coplanarity:     {info['coplanarity_metric']:.6e}")
    if 'condJ' in info:
        print(f"cond(J_damped):  {info['condJ']:.3e}")

    # --- NEW: print explicit angles in degrees ---
    if "angles_measured_deg" in info and "angles_theory_deg" in info:
        am = info["angles_measured_deg"]
        at = info["angles_theory_deg"]
        print(f"angles_measured  (deg): [{am[0]:.3f}, {am[1]:.3f}, {am[2]:.3f}]  (θ12, θ23, θ31)")
        print(f"angles_theory    (deg): [{at[0]:.3f}, {at[1]:.3f}, {at[2]:.3f}]")
    if "angle_deg_rmse" in info:
        print(f"angle_deg_rmse   (deg): {info['angle_deg_rmse']:.6f}")
    if "angles_measured_cyclic_deg" in info:
        ac = info["angles_measured_cyclic_deg"]
        print(f"angles_measured_cyclic (deg, sum=360): [{ac[0]:.3f}, {ac[1]:.3f}, {ac[2]:.3f}]")
    if "angle_deg_rmse_unweighted" in info:
        print(f"angle_deg_rmse_unweighted (deg): {info['angle_deg_rmse_unweighted']:.6f}")
    if "delta_r_unweighted" in info:
        print(f"delta_r_unweighted (len): {info['delta_r_unweighted']:.6f}")

# (No __main__ block here; this file acts as a library.)
