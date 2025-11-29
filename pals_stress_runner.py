# pals_stress_runner.py
# Stress test harness for pals_basic_v7_fix.py (numpy-only)

import time, math, json
import numpy as np
import pals_basic_v7_fix as m
from pathlib import Path

# ---------- small helpers ----------
def _sample_vertices_disk(r_src_mm, n, seed=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0*np.pi, size=n)
    rad   = r_src_mm * np.sqrt(rng.random(n))
    V = np.zeros((n,3), float)
    V[:,0] = rad * np.cos(theta)
    V[:,1] = rad * np.sin(theta)
    return V, rng

def _summ(errs):
    errs = np.array(errs, float)
    return dict(
        N_eff=int(len(errs)),
        MAE=float(errs.mean()) if len(errs) else float("nan"),
        P50=float(np.percentile(errs, 50)) if len(errs) else float("nan"),
        P90=float(np.percentile(errs, 90)) if len(errs) else float("nan"),
    )

def run_case(tag,
             N=10000, seed=42,
             det_radius=250.0, src_radius=150.0,
             noise_E_keV=0.0,        # 能量高斯噪声 σ
             noise_pos_mm=0.0,       # 每个探测点位置噪声 σ（x/y）
             z_drift_mm=0.0,         # 非共面：每个探测点 z 垂直扰动 σ
             renorm_sum_1022=True,   # 噪声后是否把能量总和归一到 1022 keV
             grid_step_mm=30.0,      # 2D 起点网格步长
             grid_radius_mm=180.0,   # 2D 起点网格半径
             polish_steps=6,         # 3D 轻抛光步数（0 关闭）
             try_all_mappings=False  # 只有在确实未知对应时才开
             ):
    """
    Returns: dict(summary + settings). Also writes tri_<tag>.csv
    """
    # 配置重建器参数（直接改模块级常量）
    m.GRID_STEP_MM   = float(grid_step_mm)
    m.GRID_RADIUS_MM = float(grid_radius_mm)
    m.POLISH_STEPS   = int(polish_steps)
    m.TRY_ALL_MAPPINGS = bool(try_all_mappings)

    R = m.build_equilateral_R(det_radius)
    V_all, rng = _sample_vertices_disk(src_radius, N, seed)

    errs, rows = [], []
    t0 = time.perf_counter()

    for V in V_all:
        # 1) 生成器：得到“真值”能量（自洽）
        try:
            E_clean = m.solve_energies_from_VR(V, R)
        except Exception:
            continue  # 病态几何/非物理解，跳过

        # 2) 构造带噪数据（不改“真值 V”）
        E = E_clean.copy()
        if noise_E_keV > 0:
            E += rng.normal(0.0, noise_E_keV, size=3)
            if renorm_sum_1022:
                s = float(E.sum())
                if s <= 1e-9:
                    continue
                E *= (1022.0 / s)
            if np.any(E <= 0.0):
                continue  # 负能量视作无效事件

        R_used = R.copy()
        if noise_pos_mm > 0 or z_drift_mm > 0:
            R_used = R_used + rng.normal(0.0, noise_pos_mm, size=R.shape)
            if z_drift_mm > 0:
                R_used[:,2] += rng.normal(0.0, z_drift_mm, size=3)  # 非共面 z 扰动

        # 3) 重建
        V_hat, score, perm = m.reconstruct_try_mappings(R_used, E)
        if V_hat is None:  # 极少数失败
            continue

        # 4) 评估
        err = float(np.linalg.norm(V_hat - V))
        errs.append(err)

        # 保存行（方便复核/画图）
        rows.append([
            V[0],V[1],V[2],
            V_hat[0],V_hat[1],V_hat[2],
            err,
            E[0],E[1],E[2],
            R_used[0,0],R_used[0,1],R_used[0,2],
            R_used[1,0],R_used[1,1],R_used[1,2],
            R_used[2,0],R_used[2,1],R_used[2,2],
            int(perm[0]),int(perm[1]),int(perm[2])
        ])

    dt = time.perf_counter() - t0
    summ = _summ(errs)
    accept = len(errs) / len(V_all) if len(V_all) else 0.0

    # 写每个场景的明细 csv
    out = Path(f"tri_{tag}.csv")
    if rows:
        arr = np.array(rows, float)
        header = (
            "Vx_true,Vy_true,Vz_true,"
            "Vx_hat,Vy_hat,Vz_hat,err_mm,"
            "E1,E2,E3,"
            "R1x,R1y,R1z,R2x,R2y,R2z,R3x,R3y,R3z,"
            "perm0,perm1,perm2"
        )
        np.savetxt(out, arr, delimiter=",", header=header, comments="")
    # 汇总信息
    result = dict(
        tag=tag,
        N_req=int(N),
        N_eff=int(summ["N_eff"]),
        accept_rate=float(accept),
        MAE_mm=float(summ["MAE"]),
        P50_mm=float(summ["P50"]),
        P90_mm=float(summ["P90"]),
        sec=float(dt),
        settings=dict(
            seed=seed, det_radius=det_radius, src_radius=src_radius,
            noise_E_keV=noise_E_keV, noise_pos_mm=noise_pos_mm,
            z_drift_mm=z_drift_mm, renorm_sum_1022=renorm_sum_1022,
            grid_step_mm=grid_step_mm, grid_radius_mm=grid_radius_mm,
            polish_steps=polish_steps, try_all_mappings=try_all_mappings
        ),
        csv=str(out) if rows else ""
    )
    print(f"[{tag}] N_eff={result['N_eff']}/{N}  "
          f"MAE={result['MAE_mm']:.3f}  P50={result['P50_mm']:.3f}  P90={result['P90_mm']:.3f}  "
          f"acc={result['accept_rate']*100:.1f}%  {result['sec']:.2f}s")
    return result

def main():
    # 测试矩阵（你可以按需增删）
    cases = [
        # 1) 无噪声（基线）
        dict(tag="baseline_N10k", N=10000, seed=42, noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=0,  try_all_mappings=False),
        dict(tag="baseline_polish", N=10000, seed=42, noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=6,  try_all_mappings=False),

        # 2) 仅能量噪声（±几 keV）
        dict(tag="E_sigma_2keV",  N=10000, seed=1,  noise_E_keV=2.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=6,  try_all_mappings=False),
        dict(tag="E_sigma_5keV",  N=10000, seed=2,  noise_E_keV=5.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=6,  try_all_mappings=False),
        dict(tag="E_sigma_10keV", N=10000, seed=3,  noise_E_keV=10.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=6,  try_all_mappings=False),

        # 3) 仅位置噪声 / 非共面
        dict(tag="R_sigma_1mm",  N=10000, seed=4, noise_E_keV=0.0, noise_pos_mm=1.0, z_drift_mm=0.0,
             grid_step_mm=20.0, polish_steps=6, try_all_mappings=False),
        dict(tag="R_sigma_3mm",  N=10000, seed=5, noise_E_keV=0.0, noise_pos_mm=3.0, z_drift_mm=0.0,
             grid_step_mm=20.0, polish_steps=6, try_all_mappings=False),
        dict(tag="Z_drift_1mm",  N=10000, seed=6, noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=1.0,
             grid_step_mm=20.0, polish_steps=6, try_all_mappings=False),

        # 4) 组合噪声
        dict(tag="E5_R1mm",  N=10000, seed=7, noise_E_keV=5.0, noise_pos_mm=1.0, z_drift_mm=0.5,
             grid_step_mm=20.0, polish_steps=8, try_all_mappings=False),
        dict(tag="E10_R3mm", N=10000, seed=8, noise_E_keV=10.0, noise_pos_mm=3.0, z_drift_mm=1.0,
             grid_step_mm=15.0, polish_steps=10, try_all_mappings=False),

        # 5) 映射未知（演示用；真实有标签时保持 False）
        dict(tag="unknown_mapping_demo", N=5000, seed=9, noise_E_keV=5.0, noise_pos_mm=1.0, z_drift_mm=0.5,
             grid_step_mm=20.0, polish_steps=8, try_all_mappings=True),
    ]

    # 执行
    report = []
    for cfg in cases:
        res = run_case(**cfg)
        res_line = dict(res)  # 复制
        # settings 写成 JSON 便于读
        res_line["settings"] = json.dumps(res_line["settings"])
        report.append(res_line)

    # 写总报表
    cols = ["tag","N_req","N_eff","accept_rate","MAE_mm","P50_mm","P90_mm","sec","settings","csv"]
    if report:
        import csv
        with open("stress_report.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in report:
                w.writerow({k:r.get(k,"") for k in cols})
        print("Saved stress_report.csv")

if __name__ == "__main__":
    main()
