# forward_generator.py
# Build "realistic-looking" events: given true V and hits R, solve E such that
#   sum(E_i * u_i) = 0  (in-plane momentum closure), and sum(E_i) = 1022 keV.

import math
import numpy as np

KEV_SUM = 1022.0

def _norm(v): 
    return float(np.linalg.norm(v))

def _unit(v):
    n = _norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def _plane_basis(R1, R2, R3):
    ex = _unit(R2 - R1)
    n  = _unit(np.cross(R2 - R1, R3 - R1))
    ey = _unit(np.cross(n, ex))
    return ex, ey, n

def solve_energies_from_VR(V, R):
    """
    Given a true vertex V and three hit points R (3x3),
    find energies E (3,) s.t.
       [u1x u2x u3x][E1]   [0]
       [u1y u2y u3y][E2] = [0]
       [  1   1   1 ][E3]   [1022]
    where (uix, uiy) are 2D components of unit directions from V to Ri on the event plane.
    Uses least squares for numerical stability. Raises if energies go negative.
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)

    ex, ey, n = _plane_basis(R[0], R[1], R[2])

    U = []
    for i in range(3):
        ui3 = _unit(R[i] - V)
        U.append(np.array([np.dot(ui3, ex), np.dot(ui3, ey)], dtype=float))
    u1, u2, u3 = U

    A = np.array([[u1[0], u2[0], u3[0]],
                  [u1[1], u2[1], u3[1]],
                  [1.0,   1.0,   1.0  ]], dtype=float)
    b = np.array([0.0, 0.0, KEV_SUM], dtype=float)

    E, *_ = np.linalg.lstsq(A, b, rcond=None)

    # physical sanity
    if np.any(E < -1e-6):
        raise ValueError(f"Non-physical energies (negative): {E}")
    E = np.maximum(E, 0.0)  # clamp tiny negative numerical junk
    return E  # keV

def sample_vertices_in_box(R, N=100, seed=0):
    """
    Uniformly sample N vertices V in a bounding box around the detector triangle.
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(R, dtype=float)
    cen = R.mean(axis=0)
    span = (R.max(axis=0) - R.min(axis=0) + 1.0)  # mm
    box_min = cen - 0.5*span
    box_max = cen + 0.5*span
    V_all = rng.uniform(box_min, box_max, size=(N,3))
    return V_all

def make_dataset(R, N=1000, noise_E_keV=0.0, noise_pos_mm=0.0, seed=0):
    """
    Build a list of synthetic events.
    Returns a dict with arrays: V_true (Nx3), E (Nx3), R_used (3x3) or (N,3,3) if noisy.
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(R, dtype=float)

    V_list, E_list = [], []
    R_list = []

    Vs = sample_vertices_in_box(R, N=N, seed=seed)
    for V in Vs:
        R_used = R + rng.normal(0.0, noise_pos_mm, size=R.shape) if noise_pos_mm>0 else R.copy()
        try:
            E = solve_energies_from_VR(V, R_used)
        except Exception:
            # skip non-physical cases
            continue

        if noise_E_keV > 0:
            E = np.clip(E + rng.normal(0.0, noise_E_keV, size=3), 0.0, 511.0)

        V_list.append(V)
        E_list.append(E)
        R_list.append(R_used)

    return {
        "V_true": np.array(V_list, dtype=float),
        "E":      np.array(E_list, dtype=float),
        "R_used": np.array(R_list, dtype=float),  # shape (M,3,3)
    }

if __name__ == "__main__":
    # Minimal smoke test
    R = np.array([[0.0, 0.0, 0.0],
                  [100.0, 0.0, 0.0],
                  [ 40.0, 80.0, 0.0]])
    V  = np.array([30.0, 20.0, 0.0])
    E  = solve_energies_from_VR(V, R)
    print("E (keV):", E, "sum:", E.sum())
