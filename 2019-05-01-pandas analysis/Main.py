import numpy as np


a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6]])
c = np.concatenate((a, a), axis=1)

print(a.shape)
print(b.shape)
print(c.shape)

