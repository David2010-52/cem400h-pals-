# pals_qc_stress.py
# 大规模压测 + 自动质控（QC）筛选 + 非共面检测
# 依赖：numpy, pals_basic_v7_fix.py（与你现在的 v7_fix 同目录）

import time, json, csv
import numpy as np
from pathlib import Path
import pals_basic_v7_fix as m

# ============ 小工具 ============

def plane_rms_mm(R):
    """三点到最佳拟合平面的RMS距离（mm）。三点时就是几何平面，带噪时能量化非共面。"""
    C = R.mean(axis=0)
    X = R - C
    # SVD: 最小奇异值方向是法向
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]
    d = np.dot(X, n)        # 每点距离（带正负）
    return float(np.sqrt(np.mean(d**2)))

def cos_triplet_from_V3(V3, R):
    U = []
    for i in range(3):
        v = R[i] - V3
        n = np.linalg.norm(v)
        if n == 0: return None
        U.append(v / n)
    u1,u2,u3 = U
    return float(np.dot(u1,u2)), float(np.dot(u1,u3)), float(np.dot(u2,u3))

def angle_rms_resid(V3, R, E):
    """用重建顶点 V3 计算到三点的 cos，与能量→角度的 cos 比较，返回RMS残差。"""
    _, (c12e,c13e,c23e) = m.energies_to_thetas_and_cos(E)
    c = cos_triplet_from_V3(V3, R)
    if c is None: return np.inf
    r = np.array([c[0]-c12e, c[1]-c13e, c[2]-c23e], float)
    return float(np.sqrt(np.mean(r**2)))

def closure_norm(V3, R, E):
    """动量闭合向量的范数 || sum(E_i * u_i) ||，单位=keV。"""
    S = m.closure_vec(V3, R, E)
    return float(np.linalg.norm(S))

def run_one_case(tag, N=10000, seed=42,
                 det_radius=250.0, src_radius=150.0,
                 noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=0.0,
                 renorm_sum_1022=True,
                 grid_step_mm=30.0, grid_radius_mm=180.0,
                 polish_steps=6, try_all_mappings=False,
                 qc=None, save_rows=True):
    """qc: dict 阈值（可为 None = 不筛选）。返回汇总信息 & CSV 路径。"""

    # 设定重建器参数
    m.GRID_STEP_MM      = float(grid_step_mm)
    m.GRID_RADIUS_MM    = float(grid_radius_mm)
    m.POLISH_STEPS      = int(polish_steps)
    m.TRY_ALL_MAPPINGS  = bool(try_all_mappings)

    # 构造探测几何 + 采样顶点
    R0 = m.build_equilateral_R(det_radius)
    def sample_vertices(n, seed):
        rng = np.random.default_rng(seed)
        th  = rng.uniform(0.0, 2*np.pi, size=n)
        r   = src_radius * np.sqrt(rng.random(n))
        V   = np.zeros((n,3), float); V[:,0]=r*np.cos(th); V[:,1]=r*np.sin(th)
        return V, rng

    V_all, rng = sample_vertices(N, seed)

    rows, errs, errs_keep = [], [], []
    qc_flags = dict(angle=0, closure=0, plane=0, sumE=0, negE=0)
    keep_cnt = 0
    t0 = time.perf_counter()

    for V in V_all:
        # 1) 生成“真值能量”（自洽）；生成器里会跳过病态几何
        try:
            E_clean = m.solve_energies_from_VR(V, R0)
        except Exception:
            continue

        # 2) 加噪
        E = E_clean.copy()
        if noise_E_keV > 0:
            E += rng.normal(0.0, noise_E_keV, size=3)
            if renorm_sum_1022:
                s = float(E.sum()); 
                if s <= 1e-9: continue
                E *= (1022.0 / s)

        if np.any(E <= 0.0):
            qc_flags["negE"] += 1
            continue

        R_used = R0.copy()
        if noise_pos_mm > 0:
            R_used += rng.normal(0.0, noise_pos_mm, size=R_used.shape)
        if z_drift_mm > 0:
            R_used[:,2] += rng.normal(0.0, z_drift_mm, size=3)

        # 3) 重建
        V_hat, score, perm = m.reconstruct_try_mappings(R_used, E)
        if V_hat is None:
            continue

        # 4) 评估与QC指标
        err   = float(np.linalg.norm(V_hat - V))
        a_rms = angle_rms_resid(V_hat, R_used, E)
        c_norm= closure_norm(V_hat, R_used, E)
        p_rms = plane_rms_mm(R_used)
        d_sum = abs(float(E.sum()) - 1022.0)

        keep = True
        if qc is not None:
            if a_rms   > qc["angle_rms_thr"]:  keep=False; qc_flags["angle"]  += 1
            if c_norm  > qc["closure_thr"]:    keep=False; qc_flags["closure"]+= 1
            if p_rms   > qc["plane_rms_thr"]:  keep=False; qc_flags["plane"]  += 1
            if d_sum   > qc["sumE_thr"]:       keep=False; qc_flags["sumE"]  += 1

        errs.append(err)
        if keep:
            keep_cnt += 1
            errs_keep.append(err)

        if save_rows:
            rows.append([
                V[0],V[1],V[2],
                V_hat[0],V_hat[1],V_hat[2],
                err, a_rms, c_norm, p_rms, d_sum,
                E[0],E[1],E[2],
                R_used[0,0],R_used[0,1],R_used[0,2],
                R_used[1,0],R_used[1,1],R_used[1,2],
                R_used[2,0],R_used[2,1],R_used[2,2],
                int(perm[0]),int(perm[1]),int(perm[2]),
                int(keep)
            ])

    dt = time.perf_counter() - t0
    errs = np.array(errs, float); errs_keep = np.array(errs_keep, float)

    def _summ(e):
        if e.size==0: return (np.nan, np.nan, np.nan)
        return (float(e.mean()), float(np.percentile(e,50)), float(np.percentile(e,90)))

    mae, p50, p90 = _summ(errs)
    mae_k, p50_k, p90_k = _summ(errs_keep)

    # 写明细
    csv_path = Path(f"triqc_{tag}.csv")
    if rows:
        header = ("Vx_true,Vy_true,Vz_true,"
                  "Vx_hat,Vy_hat,Vz_hat,err_mm,"
                  "angle_rms,closure_norm,plane_rms_mm,sumE_dev_keV,"
                  "E1,E2,E3,"
                  "R1x,R1y,R1z,R2x,R2y,R2z,R3x,R3y,R3z,"
                  "perm0,perm1,perm2,keep")
        np.savetxt(csv_path, np.array(rows, float), delimiter=",", header=header, comments="")

    res = dict(
        tag=tag, N_req=N, N_eff=len(errs), sec=dt,
        MAE=mae, P50=p50, P90=p90,
        MAE_after=mae_k, P50_after=p50_k, P90_after=p90_k,
        keep=keep_cnt, keep_rate=(keep_cnt/len(errs) if len(errs)>0 else 0.0),
        qc_flags=qc_flags, csv=str(csv_path)
    )
    print(f"[{tag}] N_eff={res['N_eff']}/{N}  "
          f"MAE={mae:.3f}  P50={p50:.3f}  P90={p90:.3f}  |  "
          f"afterQC: MAE={mae_k:.3f} P50={p50_k:.3f} P90={p90_k:.3f}  "
          f"keep={res['keep_rate']*100:.1f}%  {dt:.2f}s")
    return res

def learn_qc_thresholds(N=10000, seed=42, det_radius=250.0, src_radius=150.0,
                        grid_step_mm=30.0, grid_radius_mm=180.0, polish_steps=0):
    """在无噪声下学习 angle/closure 阈值（用99百分位×放大因子）。"""
    # 运行一次无噪声，采集分布
    tag = "baseline_learn"
    m.GRID_STEP_MM   = grid_step_mm
    m.GRID_RADIUS_MM = grid_radius_mm
    m.POLISH_STEPS   = polish_steps
    m.TRY_ALL_MAPPINGS = False

    R0 = m.build_equilateral_R(det_radius)
    th  = np.random.default_rng(seed).uniform(0, 2*np.pi, size=N)
    r   = src_radius*np.sqrt(np.random.default_rng(seed+1).random(N))
    V_all = np.zeros((N,3), float); V_all[:,0]=r*np.cos(th); V_all[:,1]=r*np.sin(th)

    angles, clos = [], []
    cnt = 0
    for V in V_all:
        try:
            E = m.solve_energies_from_VR(V, R0)
        except Exception:
            continue
        V_hat, _, _ = m.reconstruct_try_mappings(R0, E)
        if V_hat is None: continue
        angles.append(angle_rms_resid(V_hat, R0, E))
        clos.append(closure_norm(V_hat, R0, E))
        cnt += 1

    angles = np.array(angles, float); clos = np.array(clos, float)
    a_p99  = float(np.percentile(angles, 99))
    c_p99  = float(np.percentile(clos,   99))

    qc = dict(
        angle_rms_thr = max(1e-6, 10.0 * a_p99),   # 放大倍数可按需调
        closure_thr   = max(1e-6, 10.0 * c_p99),
        plane_rms_thr = 0.6,       # 非共面RMS阈值（mm），>~1mm 明显非共面
        sumE_thr      = 5.0        # 能量和偏差阈值（keV）
    )
    print(f"[QC] learned thr: angle_rms<={qc['angle_rms_thr']:.2e}, "
          f"closure<={qc['closure_thr']:.2e}, plane_rms<={qc['plane_rms_thr']:.2f} mm, "
          f"|sumE-1022|<={qc['sumE_thr']:.1f} keV")
    return qc

def main():
    # 1) 学阈值（无噪声）
    qc = learn_qc_thresholds(N=10000, seed=42)

    # 2) 定义要压测并套用QC的场景
    cases = [
        # 基线
        dict(tag="baseline", N=10000, seed=42, noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=0, try_all_mappings=False),

        # 仅能量噪声
        dict(tag="E_2keV", N=10000, seed=1, noise_E_keV=2.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=6, try_all_mappings=False),
        dict(tag="E_5keV", N=10000, seed=2, noise_E_keV=5.0, noise_pos_mm=0.0, z_drift_mm=0.0,
             grid_step_mm=30.0, polish_steps=8, try_all_mappings=False),

        # 仅位置 / 非共面
        dict(tag="R_1mm", N=10000, seed=3, noise_E_keV=0.0, noise_pos_mm=1.0, z_drift_mm=0.0,
             grid_step_mm=20.0, polish_steps=6, try_all_mappings=False),
        dict(tag="R_3mm", N=10000, seed=4, noise_E_keV=0.0, noise_pos_mm=3.0, z_drift_mm=0.0,
             grid_step_mm=20.0, polish_steps=8, try_all_mappings=False),
        dict(tag="Z_1mm", N=10000, seed=5, noise_E_keV=0.0, noise_pos_mm=0.0, z_drift_mm=1.0,
             grid_step_mm=20.0, polish_steps=8, try_all_mappings=False),

        # 组合 + 显式非共面
        dict(tag="E5_R1mm_Z05", N=10000, seed=6, noise_E_keV=5.0, noise_pos_mm=1.0, z_drift_mm=0.5,
             grid_step_mm=20.0, polish_steps=10, try_all_mappings=False),
    ]

    # 3) 跑并输出报表
    report = []
    for cfg in cases:
        res = run_one_case(**cfg, qc=qc, save_rows=True)
        res_line = dict(res); res_line["qc"] = json.dumps(qc)
        report.append(res_line)

    cols = ["tag","N_req","N_eff","keep","keep_rate","MAE","P50","P90",
            "MAE_after","P50_after","P90_after","sec","csv","qc"]
    with open("qc_report.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in report:
            w.writerow({k:r.get(k,"") for k in cols})
    print("Saved qc_report.csv")

if __name__ == "__main__":
    main()
