# run_batch_csv.py
# Purpose: Batch 3γ vertex reconstruction from CSV (offline friendly)
# Usage:
#   python run_batch_csv.py --input events.csv --output results.csv
# Options:
#   --cols "x1,y1,z1,x2,y2,z2,x3,y3,z3,e1,e2,e3"
#   --sum_tol 60 --emax 511 --no_preselect
#   --rel_mis_th 1e-3 --angle_rmse_th 1e-3 --coplanarity_th 1e-3 --condJ_th 1e3

import argparse
import csv
import numpy as np
from typing import List, Tuple
from main import solve_vertex_from_hits   # 你的库
# --- imports ---
import csv
import argparse
import numpy as np
from typing import List, Tuple

# 重要：这里要导入你刚刚改过、且会向 info 写入新字段的函数
# 如果函数在 main.py，就用这一行；如果在别的文件（比如 vertex_solver.py），就改成对应模块名
from main import solve_vertex_from_hits

# 可选：打印一下当前用的 solver 来自哪个文件，避免导错旧版本
import inspect
print("USING solver from:", inspect.getsourcefile(solve_vertex_from_hits))

DEFAULT_COLS = "x1,y1,z1,x2,y2,z2,x3,y3,z3,e1,e2,e3"

def parse_cols(spec: str) -> List[str]:
    cols = [s.strip() for s in spec.split(",")]
    if len(cols) != 12:
        raise ValueError(f"--cols expects 12 names, got {len(cols)}")
    return cols

def row_to_arrays(row: dict, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vals = []
    for name in cols:
        if name not in row:
            raise KeyError(f"Missing column '{name}'")
        try:
            vals.append(float(row[name]))
        except ValueError:
            raise ValueError(f"Bad float in column '{name}': {row[name]!r}")
    x1,y1,z1,x2,y2,z2,x3,y3,z3,e1,e2,e3 = vals
    a = np.array([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]], dtype=float)
    E = np.array([e1,e2,e3], dtype=float)
    return a, E

def preselect(E: np.ndarray, sum_tol: float, emax: float, strict: bool=True):
    reasons = []
    ok = True
    if np.any(E >= emax + 1e-9):
        ok = False; reasons.append(f"Ei>={emax}")
    if abs(float(E.sum()) - 1022.0) > sum_tol:
        ok = False; reasons.append(f"|sum-1022|>{sum_tol}")
    reason = ";".join(reasons) if reasons else "OK"
    return (ok or not strict), reason

def qc_pass(info: dict, E: np.ndarray,
            rel_mis_th: float, angle_rmse_th: float,
            coplanarity_th: float, condJ_th: float,
            angle_rmse_unw_th: float, dr_th: float) -> bool:
    rel_mis = float(info["misclosure_norm"]) / (float(E.sum()) + 1e-12)
    ok = (rel_mis < rel_mis_th and
          info.get("angle_cos_rmse", 1.0) < angle_rmse_th and
          info.get("coplanarity_metric", 1.0) < coplanarity_th and
          info.get("condJ", 1e12) < condJ_th and
          bool(info.get("converged", False)))
    if not ok:
        return False

    ang_unw = float(info.get("angle_deg_rmse_unweighted", 0.0))
    dr_unw  = float(info.get("delta_r_unweighted", 0.0))

    # 角度严重失配 -> 直接失败
    if ang_unw > angle_rmse_unw_th:
        return False

    # 角度有点偏，则需要两者同时偏大才失败（防误杀）
    if ang_unw > 1.0 and dr_unw > dr_th:
        return False

    return True



def main():
    ap = argparse.ArgumentParser(description="Batch 3γ vertex reconstruction from CSV.")
    ap.add_argument("--input", required=True, help="Input CSV with events")
    ap.add_argument("--output", required=True, help="Output CSV for results")
    ap.add_argument("--cols", default=DEFAULT_COLS,
                    help=f"Column names (default: {DEFAULT_COLS})")
    ap.add_argument("--sum_tol", type=float, default=60.0, help="|sum(E)-1022| tol (keV)")
    ap.add_argument("--emax", type=float, default=511.0, help="per-channel max (keV)")
    ap.add_argument("--no_preselect", action="store_true", help="disable preselection")
    # QC thresholds
    ap.add_argument("--rel_mis_th", type=float, default=1e-3)
    ap.add_argument("--angle_rmse_th", type=float, default=1e-3)
    ap.add_argument("--coplanarity_th", type=float, default=1e-3)
    ap.add_argument("--condJ_th", type=float, default=1e3)
    # NEW: cross-check thresholds
    ap.add_argument("--angle_rmse_unw_th", type=float, default=5.0,  # 原来 0.5
                    help="Max unweighted-vs-energy angle RMSE (deg)")
    ap.add_argument("--dr_th", type=float, default=60.0,  # 原来 5.0
                    help="Max |r(weighted)-r(unweighted)| in position units")
    args = ap.parse_args()

    cols = parse_cols(args.cols)

    # 打开文件（utf-8-sig 兼容 Excel 的 BOM）
    with open(args.input, "r", newline="", encoding="utf-8-sig") as fi, \
         open(args.output, "w", newline="", encoding="utf-8") as fo:
        reader = csv.DictReader(fi)

        # 表头（新增角度四列）
        fieldnames = (["event_index", "selected", "select_reason"] +
                      ["r_x", "r_y", "r_z",
                       "misclosure_norm", "rel_mis",
                       "angle_cos_rmse", "coplanarity", "condJ",
                       "converged", "iters","angle_deg_rmse_unweighted", "delta_r_unweighted",
                       "theta12_deg", "theta23_deg", "theta31_deg", "angle_deg_rmse","theta12_cyc_deg", "theta23_cyc_deg", "theta31_cyc_deg"])
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        n_total = n_selected = n_pass_qc = 0

        # ===================== 主循环（正确的 try/except 结构） =====================
        for idx, row in enumerate(reader, start=1):
            n_total += 1

            # 1) 解析一行（只包 row_to_arrays）
            try:
                a, E = row_to_arrays(row, cols)
            except Exception as ex:
                writer.writerow({
                    "event_index": idx,
                    "selected": False,
                    "select_reason": f"parse_error:{ex}",
                    "r_x": "", "r_y": "", "r_z": "",
                    "misclosure_norm": "", "rel_mis": "",
                    "angle_cos_rmse": "", "coplanarity": "", "condJ": "",
                    "converged": False, "iters": 0,
                    "theta12_deg": "", "theta23_deg": "", "theta31_deg": "", "angle_deg_rmse": "",
                    "theta12_cyc_deg": "", "theta23_cyc_deg": "", "theta31_cyc_deg": "",
                    "angle_deg_rmse_unweighted": "", "delta_r_unweighted": "",
                })
                continue

            # 2) 预筛
            selected, reason = preselect(E, args.sum_tol, args.emax, strict=not args.no_preselect)
            if not selected:
                writer.writerow({
                    "event_index": idx,
                    "selected": False,
                    "select_reason": reason,
                    "r_x": "", "r_y": "", "r_z": "",
                    "misclosure_norm": "", "rel_mis": "",
                    "angle_cos_rmse": "", "coplanarity": "", "condJ": "",
                    "converged": False, "iters": 0,
                    "theta12_deg": "", "theta23_deg": "", "theta31_deg": "", "angle_deg_rmse": "",
                    "theta12_cyc_deg": "", "theta23_cyc_deg": "", "theta31_cyc_deg": "",
                    "angle_deg_rmse_unweighted": "", "delta_r_unweighted": "",
                })
                continue

            n_selected += 1

            # 3) 重建（单独 try 捕获异常）
            # 3) 重建（单独 try 捕获异常）
            try:
                r, info = solve_vertex_from_hits(a, E)

                # 质量指标与是否通过 QC
                rel_mis = float(info["misclosure_norm"]) / (float(E.sum()) + 1e-12)
                passed = qc_pass(info, E,
                                 args.rel_mis_th, args.angle_rmse_th,
                                 args.coplanarity_th, args.condJ_th,
                                 args.angle_rmse_unw_th, args.dr_th)
                if passed:
                    n_pass_qc += 1

                # === 角度取值 ===
                # 两两最小夹角（你之前就有）：用于 angle_deg_rmse 等 QC
                am = info.get("angles_measured_deg", [float("nan")] * 3)
                # NEW：圆周扇区角（和=360°），来自 main.py 里我们新加的 info 字段
                ac = info.get("angles_measured_cyclic_deg", [float("nan")] * 3)

                # === 写出一行结果（把新列追加进去） ===
                writer.writerow({
                    "event_index": idx,
                    "selected": True,
                    "select_reason": reason,
                    "r_x": f"{r[0]:.9g}",
                    "r_y": f"{r[1]:.9g}",
                    "r_z": f"{r[2]:.9g}",
                    "misclosure_norm": f"{info['misclosure_norm']:.6e}",
                    "rel_mis": f"{rel_mis:.6e}",
                    "angle_cos_rmse": f"{info['angle_cos_rmse']:.6e}",
                    "coplanarity": f"{info['coplanarity_metric']:.6e}",
                    "condJ": f"{info['condJ']:.6e}",
                    "converged": info["converged"],
                    "iters": info["iters"],

                    # 已有：三对两两最小夹角 + 角度RMSE
                    "theta12_deg": f"{am[0]:.6f}",
                    "theta23_deg": f"{am[1]:.6f}",
                    "theta31_deg": f"{am[2]:.6f}",
                    "angle_deg_rmse": f"{info.get('angle_deg_rmse', float('nan')):.6f}",
                    # NEW：等权交叉验证与两解距离
                    "angle_deg_rmse_unweighted": f"{info.get('angle_deg_rmse_unweighted', float('nan')):.6f}",
                    "delta_r_unweighted": f"{info.get('delta_r_unweighted', float('nan')):.6f}",
                    # NEW：三段“圆周扇区角”（必和=360°）
                    "theta12_cyc_deg": f"{ac[0]:.6f}",
                    "theta23_cyc_deg": f"{ac[1]:.6f}",
                    "theta31_cyc_deg": f"{ac[2]:.6f}",
                    "angles_cyc_sum_deg": f"{float(np.sum(ac)):.6f}",
                })

            except Exception as ex:
                # （这个分支保持你原来写法，但要包含三列扇区角，留空）
                writer.writerow({
                    "event_index": idx,
                    "selected": True,
                    "select_reason": f"recon_error:{ex}",
                    "r_x": "", "r_y": "", "r_z": "",
                    "misclosure_norm": "", "rel_mis": "",
                    "angle_cos_rmse": "", "coplanarity": "", "condJ": "",
                    "converged": False, "iters": 0,
                    "theta12_deg": "", "theta23_deg": "", "theta31_deg": "", "angle_deg_rmse": "",
                    # NEW 扇区角三列也要有（留空）
                    "theta12_cyc_deg": "", "theta23_cyc_deg": "", "theta31_cyc_deg": "",
                    "angle_deg_rmse_unweighted": "", "delta_r_unweighted": "",
                })

        # =====================================================================

        print(f"Done. Total rows: {n_total}, selected: {n_selected}, passed QC: {n_pass_qc}")
        print(f"Results written to: {args.output}")

if __name__ == "__main__":
    main()
