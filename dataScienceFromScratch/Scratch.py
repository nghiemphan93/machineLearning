import random
from typing import List

Vector = List[float]




def mul(vectorA: Vector, scalar: float) -> Vector:
   return [temp * scalar for temp in vectorA]


def add(vectorA: Vector, vectorB: Vector) -> Vector:
   return [a + b for a, b in zip(vectorA, vectorB)]


def sumOfVector(vectorA: Vector) -> float:
   return sum(vectorA)


def dot(vectorA: Vector, vectorB: Vector) -> float:
   product = [a * b for a, b in zip(vectorA, vectorB)]
   return sum(product)


def vectorSum(vectors: List[Vector]) -> Vector:
   numbElement = len(vectors[0])
   return [sum(vector[i] for vector in vectors)
           for i in range(numbElement)]


def vectorMean(vectors: List[Vector]) -> Vector:
   return mul(vectorSum(vectors), 1 / len(vectors))


vectorA = [1, 2, 3]
vectorB = [4, 5, 6]
vectors = [[1, 2, 3],
           [4, 5, 6]]

assert add(vectorA, vectorB) == [5, 7, 9]
assert mul(vectorA, 2) == [2, 4, 6]
assert sumOfVector(vectorA) == 6
assert dot(vectorA, vectorA) == 14
assert vectorSum(vectors) == [5, 7, 9]
assert vectorMean(vectors) == [2.5, 3.5, 4.5]


def updateWeights(weights: Vector, gradient: Vector, learningRate: float = -0.001) -> Vector:
   return add(weights, mul(gradient, learningRate))


def sumOfSquareGradients(vectorA: Vector) -> Vector:
   return [a * 2 for a in vectorA]


def linearGradient(x: float, y: float, weights: Vector) -> Vector:
   yPredicted = weights[0] * x + weights[1]
   loss = yPredicted - y
   gradient = [2 * loss * x, 2 * loss * 1]
   return gradient


vectorB = [452.19603984, 314.63722789, 315.46835808]
# for i in range(2000):
#    vectorB = updateWeights(weights=vectorB, gradient=sumOfSquareGradients(vectorB))
#    print(vectorB)

X: Vector = [i for i in range(-50, 50)]
Y: Vector = [20 * x + 5 for x in X]
weights = [random.random(), random.random()]
print(weights)
for i in range(1000):
   meanGradient = vectorMean([linearGradient(x, y, weights) for x, y in zip(X, Y)])
   # print(meanGradient)
   weights = updateWeights(weights=weights, gradient=meanGradient)
   print(weights)
0