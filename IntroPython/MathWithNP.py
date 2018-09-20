import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
sinx = np.sin(x)
cosx = np.cos(x)

print(sinx)
plt.plot(x, sinx, "x")
plt.plot(x, cosx, "o")
plt.show()