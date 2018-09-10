import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1,2,4,3,5])
y = np.array([1,3,3,2,5])

# Optimize a, b in  yHat = a*x + b
def optimizeAB(X, y):
    a = b = 0.0
    alpha = 0.01
    # 10 Iterations though data set
    for j in range (10):
        for i in range(len(X)):
            a, b = updateAB(X[i], y[i], a, b, alpha)
    return a, b

# Update value for next a, b, c
def updateAB(x, y, a, b, alpha):
    yHat = a * x + b
    error = yHat - y
    a = a - alpha * error * x
    b = b - alpha * error
    return a, b


'''==================================================='''
# Show test results
a, b = optimizeAB(X, y)
yHat = [a*x + b for x in X]
testX = 4.5
predictY = a * testX + b

plt.scatter(X, y)
plt.scatter(testX, predictY, color="brown")
plt.plot(X, yHat)
plt.show()

