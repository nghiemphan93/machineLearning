from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from numpy import float64


# Calculate the best fit line y = a*x + b, given all data pairs
def bestFitLine(X, y):
    a = ((mean(X) * mean(y)) - mean(X*y)) / (mean(X)**2 - mean(X**2))
    b = mean(y) - a * mean(X)
    return a, b

# Sum of square of difference of each original y and yHat (y from the best fit line)
def squareError(y, yHat):
    return sum((y - yHat)**2)

# Number determines how fit the line to the given data pairs
def correlationCoefficient(y, yHat):
    yRegressionMean = [mean(y) for _y in yHat]
    squareErrorYHat = squareError(y, yHat)
    squareErrorYMean = squareError(y, yRegressionMean)
    return 1 - squareErrorYHat/ squareErrorYMean



# Demo
# Height (cm)
X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
# Weight (kg)
y = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# y = a*x + b
a, b = bestFitLine(X, y)


yHat = [a*x + b for x in X]

testX = 190
predictY = a * testX + b

plt.scatter(X, y)
plt.scatter(testX, predictY, color="brown")
plt.plot(X, yHat)
plt.show()

print(correlationCoefficient(y, yHat))
