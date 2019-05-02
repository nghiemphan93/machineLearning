import numpy as np

a = np.arange(10)
b = np.where(a < 5, a, 10 * a)
a[np.nonzero(a >= 5)] = a[np.nonzero(a >= 5)] * 10

print(a)
print(b)
print(a)
