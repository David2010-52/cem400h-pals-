# gen_synth_events.py
# Generate synthetic 3γ events for testing (clean / noisy / mismatch).
# Usage examples:
#   python gen_synth_events.py --mode clean --n 50 --out events_clean.csv
#   python gen_synth_events.py --mode noise --n 50 --out events_noise.csv
#   python gen_synth_events.py --mode mismatch --n 20 --out events_mismatch.csv

import argparse, math, numpy as np, csv, random

def angles_from_energies(E):
    E1,E2,E3 = map(float,E)
    c12 = (E3**2 - E1**2 - E2**2) / (2*E1*E2)
    c23 = (E1**2 - E2**2 - E3**2) / (2*E2*E3)
    c31 = (E2**2 - E3**2 - E1**2) / (2*E3*E1)
    cosines = np.clip([c12,c23,c31], -1.0, 1.0)
    ang = np.degrees(np.arccos(cosines))
    return ang  # [θ12, θ23, θ31], sum≈360

def energies_from_angles(theta12, theta23, theta31):
    # law of sines for the "energy triangle": E1: sin(theta23), E2: sin(theta31), E3: sin(theta12)
    s1, s2, s3 = math.sin(math.radians(theta23)), math.sin(math.radians(theta31)), math.sin(math.radians(theta12))
    v = np.array([s1, s2, s3], dtype=float)
    E = 1022.0 * v / v.sum()
    return E

def random_plane_basis(rng):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    a = rng.normal(size=3); a -= a.dot(n)*n; a /= np.linalg.norm(a)
    b = np.cross(n, a); b /= np.linalg.norm(b)
    return a, b, n

def make_one_event(rng, mode="clean"):
    # sample two angles, third = 360 - sum; keep them away from extremes
    for _ in range(1000):
        t1 = rng.uniform(90, 150)
        t2 = rng.uniform(90, 150)
        t3 = 360.0 - t1 - t2
        if 80.0 < t3 < 150.0:
            E = energies_from_angles(t1, t2, t3)   # sum=1022
            if np.all(E < 511.0):
                break
    else:
        raise RuntimeError("Failed to sample angles/energies under constraints")

    # unit directions in a random plane with those cyclic angles
    a_hat, b_hat, n_hat = random_plane_basis(rng)
    phi1, phi2, phi3 = 0.0, t1, t1 + t2  # cyclic sectors that sum to 360
    u = np.array([
        math.cos(math.radians(phi1))*a_hat + math.sin(math.radians(phi1))*b_hat,
        math.cos(math.radians(phi2))*a_hat + math.sin(math.radians(phi2))*b_hat,
        math.cos(math.radians(phi3))*a_hat + math.sin(math.radians(phi3))*b_hat,
    ])
    # vertex inside a small volume
    r0 = np.array([rng.uniform(-30,30), rng.uniform(-30,30), rng.uniform(-30,30)])
    # hit distances (to emulate detector)
    d = rng.uniform(120, 220, size=3)
    a_pts = r0 + u * d[:,None]

    if mode == "noise":
        # add small measurement noise
        a_pts += rng.normal(scale=0.3, size=a_pts.shape)  # ~0.3 mm
        E = E + rng.normal(scale=2.0, size=3)             # ~2 keV noise
        E = np.clip(E, 10.0, 510.0)
        E *= 1022.0 / E.sum()
    elif mode == "mismatch":
        # rotate u2 by +5 degrees within plane -> geometry-energy mismatch
        rot = 5.0
        phi2m = phi2 + rot
        u[1] = math.cos(math.radians(phi2m))*a_hat + math.sin(math.radians(phi2m))*b_hat
        a_pts = r0 + u * d[:,None]
    elif mode == "oop":
        # push hit-2 out of the plane by ~5 mm
        a_pts = r0 + u * d[:, None]
        a_pts[1] = a_pts[1] + 5.0 * n_hat

    # return row: x1..z3,e1..e3
    (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) = a_pts
    e1,e2,e3 = E
    return [x1,y1,z1,x2,y2,z2,x3,y3,z3,e1,e2,e3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clean","noise","mismatch","oop"], default="clean")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x1","y1","z1","x2","y2","z2","x3","y3","z3","e1","e2","e3"])
        for _ in range(args.n):
            w.writerow(make_one_event(rng, args.mode))
    print(f"Wrote {args.n} rows to {args.out} (mode={args.mode})")

if __name__ == "__main__":
    main()
