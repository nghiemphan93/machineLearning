from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Demo

X = np.array([1,2,4,3,5])
y = np.array([1,3,3,2,5])


# Optimize a, b in  yHat = a*x + b
def optimizeAB(X, y):
    a = 0.0
    b = 0.0
    alpha = 0.01

    # 10 Iterations though data set
    for j in range (10):
        for i in range(len(X)):
            yHat = a*X[i] + b
            error = yHat - y[i]
            b = b - alpha*error
            a = a - alpha*error*X[i]
            print(a, b)
    return a, b


a, b = optimizeAB(X, y)

yHat = [a*x + b for x in X]

testX = 4.5
predictY = a * testX + b

plt.scatter(X, y)
plt.scatter(testX, predictY, color="brown")
plt.plot(X, yHat)
plt.show()

