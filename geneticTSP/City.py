import math


class City:
   """
   Representation of each place to visit
   """

   def __init__(self, x: float, y: float):
      self.x = x
      self.y = y

   def distanceTo(self, nextCity) -> float:
      """
      Calculate the euclid distance between two cities
      :param nextCity: City
      :return:
      """
      xDis = abs(self.x - nextCity.x)
      yDis = abs(self.y - nextCity.y)
      distance = math.sqrt((xDis ** 2) + (yDis ** 2))
      return distance

   def __repr__(self):
      return "(" + str(self.x) + "," + str(self.y) + ")"
