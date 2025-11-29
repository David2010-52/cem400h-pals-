# better_plot_errs.py
import numpy as np, matplotlib.pyplot as plt

a   = np.loadtxt("tri_results.csv", delimiter=",", skiprows=1)
err = a[:,6]
print("N=",len(err),"MAE=",err.mean(),"P50=",np.percentile(err,50),"P90=",np.percentile(err,90))

# 只对 >0 的误差做对数直方图（0 会在 log 下出问题）
pos = err[err > 0]
hi  = np.percentile(pos, 99.9)   # 切掉极少数尾部
lo  = pos.min()

bins = np.geomspace(lo, hi, 60)
plt.hist(pos, bins=bins)
plt.xscale('log')
plt.xlabel("error (mm, log scale)")
plt.ylabel("count"); plt.title("Vertex error (log-x)")
# 可视化 P50/P90
p50, p90 = np.percentile(pos, 50), np.percentile(pos, 90)
plt.axvline(p50, ls="--"); plt.axvline(p90, ls="--")
plt.tight_layout(); plt.show()
