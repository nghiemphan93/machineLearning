import numpy as np
import matplotlib.pyplot as plt


a = np.random.standard_normal((2,4))
b = np.random.standard_normal((5,4))
c = np.vstack([a, b])
#np.save("c", c)

c = np.load("c.npy")

print(c)
print(c.argmax(axis=0))