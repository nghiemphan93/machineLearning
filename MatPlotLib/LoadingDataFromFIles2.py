import matplotlib.pyplot as plt
import numpy as np
import csv

x, y = np.loadtxt("example", delimiter=",", unpack=True)
print(x)


plt.plot(x, y, label="Loaded from file")

plt.xlabel("Plot Number")
plt.ylabel("Important var")
plt.title("Interesting Graph\nCheck it out")
plt.legend()

plt.show()
