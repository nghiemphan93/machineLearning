import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)


x = np.arange(0,100)
y = x*2
z = x**2
'''
fig: plt.Figure = plt.figure()
axes: plt.Axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2: plt.Axes = fig.add_axes([0.15, 0.6, 0.4, 0.3])
axes.plot(x, y)
axes.set_title("Larger Plot")
axes.set_xlabel("X label")
axes.set_ylabel("Y label")
axes2.plot(y, x, "red")
plt.show()
'''
'''
fig, axes = plt.subplots(nrows=1, ncols=2)
for ax in axes:
   ax.plot(x, y)
plt.tight_layout()
plt.show()
'''
'''
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.plot(x, y, label="X Squared")
axes.plot(y, x, label="Square Root of X")
axes.legend(loc=0)
plt.tight_layout()
#fig.savefig("clgt.png", dpi=300)
plt.show()
'''
'''
fig: plt.Figure = plt.figure(dpi=300)
ax: plt.Axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y, color="b", ls="--", marker="o", markersize=10, markerfacecolor="y")
plt.show()
'''
'''
fig: plt.Figure = plt.figure(dpi=300)
ax: plt.Axes = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.plot(x, y)
ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
fig.legend()
plt.show()
'''
