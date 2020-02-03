import numpy as np
import scipy.linalg as la

A = np.asarray([
   [-1, 1, 0, 0],
   [-1, 0, 1, 0],
   [0, -1, 1, 0],
   [0, -1, 0, 1],
   [0, 0, -1, 1]
])

B = np.asarray([
   [1, 1, 1],
   [1, 1, -1],
   [-1, -1, 1]
])

C = np.array([
   [1, 0],
   [1, 1],
   [1, 2]
])

b = np.array([
   [6],
   [0],
   [0]
])

AA = np.array([
   [0, 1, 0, 0],
   [0, 0, 2, 0],
   [0, 0, 0, 3],
   [0, 0, 0, 0],
])
# U, E, V = np.linalg.svd(AA)

BB = np.array([
   [3, -4, 7, 1, -4, -3],
   [7, -6, 8, -1, -1, -7]
])
U, E, V = np.linalg.svd(BB)
print(U)
print(E)
print(V)


class Parent:
   def toString(self):
      return 'string'


class AnotherParent:
   def toClgt(self):
      return 'string'


class Person(Parent, AnotherParent):
   adresse = 'jiawoef'

   def __init__(self, name: str, age: int):
      self.name = name
      self.age = age

   def getName(self) -> str:
      return self.name
