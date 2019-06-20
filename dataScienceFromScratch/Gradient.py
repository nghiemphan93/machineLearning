from typing import Callable, List

import numpy as np


def sumOfSquare(vector: np.ndarray) -> float:
   return vector.dot(vector)


def differenceQuotient(f: Callable[[float], float], x: float, h: float) -> float:
   return (f(x + h) - f(x)) / h


def square(x: float) -> float:
   return x ** 2


def devOfSquare(x: float) -> float:
   return 2 * x


def sumOfSquareGradients(vector: np.ndarray) -> np.ndarray:
   return 2 * vector


vector = np.array([1, 2, 3])
vectorB = np.array([452.19603984, 314.63722789, 315.46835808])

# for i in [-1, -2, -3, -4]:
#    print(f'Square: {square(i)}, Dev: {devOfSquare(i)}, Est: {differenceQuotient(f=square, x=i, h=0.001):.4f}')


# for i in range(2000):
#    vectorB = gradientStep(vector=vectorB, gradient=sumOfSquareGradients(vectorB), learningRate=0.01)
#    print(vectorB)


inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
X = np.array([i for i in range(-50, 50)])
Y = np.array([20 * x + 5 for x in X])


def linearGradient(x: float, y: float, weights: np.ndarray) -> np.ndarray:
   yPredicted = weights[0] * x + weights[1]
   loss = yPredicted - y
   gradient = np.array([2 * loss * x, 2 * loss * 1])
   return gradient


def updateWeights(weights: np.ndarray, gradient: np.ndarray, learningRate: float = 0.01) -> np.ndarray:
   return weights - learningRate * gradient


weights = np.array([np.random.random(), np.random.random()])
print(weights)
for i in range(2000):
   # gradientMean = np.array(linearGradient(x=x, y=y, weights=weights).tolist() for x, y in zip(X, Y)).mean(axis=1)
   gradientMean = linearGradient(x=X[0], y=Y[0], weights=weights)
   weights = updateWeights(weights=weights, gradient=gradientMean, learningRate=0.01)
   print(weights)
