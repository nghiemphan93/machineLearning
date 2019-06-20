import math

from typing import List, Tuple, Callable

Vector = List[float]


def add(vectorA: Vector, vectorB: Vector):
   assert len(vectorA) == len(vectorB)
   return [a + b for a, b in zip(vectorA, vectorB)]


def subtract(vectorA: Vector, vectorB: Vector):
   assert len(vectorA) == len(vectorB)
   return [a - b for a, b in zip(vectorA, vectorB)]


def scalarMultiply(constant: float, vector: Vector):
   return [constant * v for v in vector]


def dot(vectorA: Vector, vectorB: Vector):
   assert len(vectorA) == len(vectorB)
   return sum([a * b for a, b, in zip(vectorA, vectorB)])


def sumOfSquare(vectorA: Vector):
   return dot(vectorA, vectorA)


def magnitude(vectorA: Vector):
   return math.sqrt(sumOfSquare(vectorA))


def squaredDistance(vectorA: Vector, vectorB: Vector):
   assert len(vectorA) == len(vectorB)
   return sumOfSquare(subtract(vectorA, vectorB))


def distance(vectorA: Vector, vectorB: Vector):
   assert len(vectorA) == len(vectorB)
   return math.sqrt(squaredDistance(vectorA, vectorB))


vectorA = [1, 2, 3]
vectorB = [4, 5, 6]
assert add(vectorA, vectorB) == [5, 7, 9]
assert subtract(vectorA, vectorB) == [-3, -3, -3]
assert scalarMultiply(2, vectorA) == [2, 4, 6]
assert dot(vectorA, vectorB) == 32
assert sumOfSquare(vectorA) == 14
assert magnitude([3, 4]) == 5
assert squaredDistance(vectorA, vectorB) == 27
assert distance(vectorA, vectorB) == math.sqrt(27)

Matrix = List[List[float]]


def shape(matrixA: Matrix) -> Tuple[int, int]:
   numbRows = len(matrixA)
   numbCols = len(matrixA[0]) if matrixA else 0
   return numbRows, numbCols


def getRow(matrixA: Matrix, rowI: int) -> Vector:
   return matrixA[rowI]


def getColumn(matrixA: Matrix, columnI: int) -> Vector:
   return [row[columnI] for row in matrixA]


def makeMatrix(numbRows: int, numbCols: int, entryFunction: Callable[[int, int], float]) -> Matrix:
   return [[entryFunction(row, col) for row in range(numbCols)]
           for col in range(numbRows)]


def identityMatrix(n: int) -> Matrix:
   return makeMatrix(n, n, lambda i, j: 1 if i == j else 0)


def printMatrix(matrixA: Matrix):
   for row in matrixA:
      print(row)


matrixA = [[1, 2, 3],
           [4, 5, 6]]
matrixB = [[7, 8, 9],
           [10, 11, 12]]
assert shape(matrixA) == (2, 3)
assert getRow(matrixA, 0) == [1, 2, 3]
assert getColumn(matrixA, 0) == [1, 4]
assert makeMatrix(2, 3, lambda i, j: 1) == [[1, 1, 1],
                                            [1, 1, 1]]
assert identityMatrix(3) == [[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]
