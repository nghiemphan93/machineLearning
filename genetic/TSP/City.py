import numpy as np
import math

class City:
   def __init__(self, x: float, y: float):
      self.x = x
      self.y = y

   def calcDistance(self, nextCity) -> float:
      xDis = abs(self.x - nextCity.x)
      yDis = abs(self.y - nextCity.y)
      distance = math.sqrt((xDis ** 2) + (yDis ** 2))
      return distance

   def __repr__(self):
      return "(" + str(self.x) + "," + str(self.y) + ")"