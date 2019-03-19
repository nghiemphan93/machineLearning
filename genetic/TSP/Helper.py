from random import Random
from genetic.TSP.City import City

class Helper:
   def __init__(self):
      pass

   @staticmethod
   def createRoute(numbCities: int):
      random = Random()
      cityRoute = []
      for i in range(numbCities):
         cityRoute.append(City(x=random.randint(0, 500),
                               y=random.randint(0, 500)))
      return cityRoute