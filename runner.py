# runner.py  (just a tiny example to connect generator -> triangulation)
import numpy as np
from forward_generator import make_dataset
from triangle_reconstruct import reconstruct_vertex

R = np.array([[0.0, 0.0, 0.0],
              [100.0, 0.0, 0.0],
              [ 40.0, 80.0, 0.0]])

data = make_dataset(R, N=100, noise_E_keV=0.0, noise_pos_mm=0.0, seed=42)
V_true_all, E_all, R_all = data["V_true"], data["E"], data["R_used"]

errs = []
for V_true, E, R_used in zip(V_true_all, E_all, R_all):
    out = reconstruct_vertex(R_used, E, try_all_mappings=True)
    V_hat = out["V"]
    err = np.linalg.norm(V_hat - V_true)
    errs.append(err)

errs = np.array(errs)
print("MAE (mm):", errs.mean(), "P50:", np.percentile(errs, 50), "P90:", np.percentile(errs, 90))
