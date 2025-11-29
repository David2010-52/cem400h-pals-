# better_plot_errs.py
import numpy as np
import matplotlib.pyplot as plt

# 读取误差列
a   = np.loadtxt("tri_results.csv", delimiter=",", skiprows=1)
err = a[:, 6]

# 基本统计
N   = len(err)
MAE = float(err.mean())
P50 = float(np.percentile(err, 50))
P90 = float(np.percentile(err, 90))
print(f"N={N} MAE={MAE} P50={P50} P90={P90}")

# 只对 >0 的误差做对数直方图（log(0) 会出问题）
pos = err[err > 0]
if pos.size == 0:
    raise SystemExit("No positive errors to plot.")

# 切掉极少数尾部以便视觉更稳定
lo = pos.min()
hi = float(np.percentile(pos, 99.9))
if hi <= lo:  # 防止全部相等导致 bins 出问题
    hi = lo * 1.001

# 绘图
bins = np.geomspace(lo, hi, 60)
counts, edges, _ = plt.hist(pos, bins=bins)
plt.xscale("log")
plt.xlabel("error (mm, log scale)")
plt.ylabel("count")
plt.title("Vertex error (log-x)")

# 可视化 P50/P90（基于正误差）
p50 = float(np.percentile(pos, 50))
p90 = float(np.percentile(pos, 90))
ax  = plt.gca()
ylim = ax.get_ylim()
ax.axvline(p50, ls="--")
ax.axvline(p90, ls="--")
ax.text(p50, ylim[1]*0.90, f"P50={p50:.1e}", ha="right", va="top", rotation=90)
ax.text(p90, ylim[1]*0.90, f"P90={p90:.1e}", ha="right", va="top", rotation=90)

# 左上角标注 N 和 MAE
ax.text(edges[0], ylim[1]*0.98, f"N={N}   MAE={MAE:.1e}", va="top")

plt.tight_layout()
plt.savefig("vertex_error_logx.png", dpi=200)
plt.show()
print("Saved vertex_error_logx.png")
