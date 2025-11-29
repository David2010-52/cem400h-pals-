# triangle_reconstruct.py
# Reconstruct 3γ vertex V from three hit points R and energies E
# Pure triangle math + momentum-closure to pick the right mirror / mapping.

import math
import numpy as np
from itertools import permutations

def _norm(v): 
    return float(np.linalg.norm(v))

def _unit(v):
    n = _norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def _plane_basis(R1, R2, R3):
    """Build an orthonormal basis (ex, ey) on the event plane defined by R1,R2,R3."""
    ex = _unit(R2 - R1)
    n  = _unit(np.cross(R2 - R1, R3 - R1))   # plane normal
    ey = _unit(np.cross(n, ex))
    return ex, ey, n

def _to_2d(P, R1, ex, ey):
    v = P - R1
    return np.array([np.dot(v, ex), np.dot(v, ey)], dtype=float)

def _from_2d(xy, R1, ex, ey):
    return R1 + xy[0]*ex + xy[1]*ey

def _oriented_angle(u, v):
    """2D oriented angle atan2(det, dot) in (-pi, pi]."""
    det = u[0]*v[1] - u[1]*v[0]
    dot = u[0]*v[0] + u[1]*v[1]
    return math.atan2(det, dot)

def energies_to_thetas(E):
    """Convert photon energies to apex angles at the vertex using the closed form."""
    E1, E2, E3 = map(float, E)
    def cc(a,b,c):
        val = (c*c - a*a - b*b) / (2.0*a*b)
        return float(np.clip(val, -1.0, 1.0))
    th12 = math.acos(cc(E1,E2,E3))
    th13 = math.acos(cc(E1,E3,E2))
    th23 = math.acos(cc(E2,E3,E1))
    return th12, th13, th23  # radians

def _closure_error(V3d, R, E):
    """Momentum-closure residual: || sum_i Ei * (Ri - V)/||Ri-V|| ||."""
    s = np.zeros(3)
    for Ri, Ei in zip(R, E):
        ray = Ri - V3d
        nr = _norm(ray)
        if nr == 0:
            return np.inf
        # 正确写法：Ei 乘以单位方向向量
        s += Ei * (ray / nr)
    return _norm(s)


def reconstruct_vertex(R, E, try_all_mappings=True):
    """
    Inputs:
      R: (3,3) array of hit positions [mm], assumed coplanar.
      E: (3,)  energies [keV].
    Returns:
      dict with keys: V (3,), eps (closure), mapping (perm), mirror (+1/-1),
                      thetas (tuple), alpha, beta
    """
    R = np.asarray(R, dtype=float)
    E = np.asarray(E, dtype=float)

    # 2D reduction on the event plane
    ex, ey, n = _plane_basis(R[0], R[1], R[2])
    r1 = np.array([0.0, 0.0])
    r2 = np.array([_norm(R[1]-R[0]), 0.0])
    r3 = _to_2d(R[2], R[0], ex, ey)

    L12 = r2[0]
    L13 = _norm(r3 - r1)
    phi = _oriented_angle(r2 - r1, r3 - r1)

    best = None
    perms = permutations(range(3)) if try_all_mappings else [tuple(range(3))]
    for p in perms:
        Em = E[list(p)]
        th12, th13, th23 = energies_to_thetas(Em)

        s12, s13 = math.sin(th12), math.sin(th13)
        if abs(s12) < 1e-10 or abs(s13) < 1e-10:
            continue  # degenerate angle geometry

        A = L12 / s12
        B = L13 / s13

        # Closed-form for alpha at R1 apex
        y_num = A*math.sin(th12) - B*math.sin(th13 + phi)
        x_den = A*math.cos(th12) - B*math.cos(th13 + phi)
        alpha = math.atan2(y_num, x_den)
        beta  = phi - alpha

        # Three vertex edges from triangle relations
        x1 = abs(A * math.sin(th12 + alpha))
        x2 = abs(L12 * math.sin(alpha) / s12)

        # Vertex coordinates in 2D
        xx = (L12*L12 + x1*x1 - x2*x2) / (2.0*L12)
        yy = max(x1*x1 - xx*xx, 0.0) ** 0.5

        for sgn in (+1.0, -1.0):  # mirror choice
            V2 = np.array([xx, sgn*yy])
            V3 = _from_2d(V2, R[0], ex, ey)
            eps = _closure_error(V3, R, Em)
            cand = dict(V=V3, eps=eps, mapping=p, mirror=(+1 if sgn>0 else -1),
                        thetas=(th12, th13, th23), alpha=alpha, beta=beta)
            if best is None or eps < best['eps']:
                best = cand

    if best is None:
        raise RuntimeError("Reconstruction failed (degenerate geometry or invalid angles).")
    return best

if __name__ == "__main__":
    # Minimal smoke test (numbers are placeholders)
    R = np.array([[0.0, 0.0, 0.0],
                  [100.0, 0.0, 0.0],
                  [ 40.0, 80.0, 0.0]])
    E = np.array([300.0, 400.0, 322.0])  # just an example; use generator to get realistic energies
    out = reconstruct_vertex(R, E, try_all_mappings=True)
    print("V_hat:", out["V"], "closure:", out["eps"])
