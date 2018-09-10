import numpy as np
import matplotlib.pyplot as plt

# Data
X1 = np.array([2.7810836, 1.465489372, 3.396561688, 1.38807019, 3.06407232, 7.627531214, 5.332441248, 6.922596716, 8.675418651, 7.673756466])
X2 = np.array([2.550537003, 2.362125076, 4.400293529, 1.850220317, 3.005305973, 2.759262235, 2.088626775, 1.77106367, -0.242068655, 3.508563011])
y = np.array([0,0,0,0,0,1,1,1,1,1])

# Optimize a, b, c in  yHat = a*x1 + b*x2 + c
def optimizeABC(X1, X2, y):
    a = b = c = 0
    alpha = 0.3
    # 10 Iterations though data set
    for j in range(10):
        for i in range(len(X1)):
            a, b, c = updateABC(a, b, c, X1[i], X2[i], y[i], alpha)
    return a, b, c

# Update value for next a, b, c
def updateABC(a, b, c, x1, x2, y, alpha):
    yHat = 1 / (1 + np.exp(-(a * x1 + b * x2 + c)))
    a = a + alpha * (y - yHat) * yHat * (1 - yHat) * x1
    b = b + alpha * (y - yHat) * yHat * (1 - yHat) * x2
    c = c + alpha * (y - yHat) * yHat * (1 - yHat) * 1
    return a, b, c

# Return probability of the label:
# if <0.5 => label = 1
# else => label = 0
def predict(x1, x2, a, b, c):
    return 1 / (1 + np.exp(-(a*x1 + b*x2 + c)))


'''==================================================='''
# Show test results
a, b, c = optimizeABC(X1, X2, y)
print(predict(6.2, 1.5, a, b, c))

plt.scatter(6.2, 1.5, color="green")
for i in range(len(y)):
    if(y[i] == 0):
        plt.scatter(X1[i], X2[i], color="brown")
    else:
        plt.scatter(X1[i], X2[i], color="blue")
plt.show()

