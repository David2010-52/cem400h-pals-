import numpy as np, matplotlib.pyplot as plt
a = np.loadtxt("tri_results.csv", delimiter=",", skiprows=1)
err = a[:,6]
print("N=",len(err), "MAE=",err.mean(), "P50=",np.percentile(err,50), "P90=",np.percentile(err,90))
plt.hist(err, bins=50)
plt.xlabel("error (mm)"); plt.ylabel("count"); plt.title("Vertex error")
plt.show()
