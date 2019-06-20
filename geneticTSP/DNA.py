
import random
from random import Random
from typing import List

from geneticTSP.City import City


class DNA:
   """
   Representation of one combination of all cities
   """

   def __init__(self, route: List[City]):
      self.random = Random()
      self.route: List[City] = route.copy()
      self.route = self.shuffleRoute()
      self.fitness = self.calcFitness()
      self.distances = self.calcTotalDist()

   def shuffleRoute(self) -> List[City]:
      """
      Shuffle the city list to make a new combination

      :return: newRoute: List[City]
      """
      newRandomRoute: List[City] = self.route[1:]
      self.random.shuffle(newRandomRoute)
      newRandomRoute.insert(0, self.route[0])
      return newRandomRoute

   def calcFitness(self) -> float:
      """
      Calculate fitness inversely proportional according to distance between all cities

      :return: self.fitness: float
      """
      totalDistance = self.calcTotalDist()
      self.fitness = 1 / totalDistance
      return self.fitness

   def calcTotalDist(self) -> float:
      """
      Total distances between all cities

      :return: totalDist: float
      """
      totalDist = 0.0
      for i in range(len(self.route) - 1):
         totalDist = totalDist + self.route[i].distanceTo(self.route[i + 1])
      totalDist = totalDist + self.route[-1].distanceTo(self.route[0])
      self.distances = totalDist
      return totalDist

   def crossover(self, partner: 'DNA') -> 'DNA':
      """
      Take two parents then combine to make a new child

      :param partner: 'DNA'
      :return: child: 'DNA'
      """
      routeA = self.route
      routeB = partner.route
      midpoint = int(random.randint(0, len(routeA) - 2))
      newRoute = routeA[0:midpoint]
      for i in range(len(routeB)):
         if routeB[i] not in newRoute:
            newRoute.append(routeB[i])
      child = DNA(route=newRoute)
      child.route = newRoute
      return child

   def mutate(self, mutationRate: float) -> None:
      """
      Randomly reorganize the route to create diversity

      :param mutationRate: float
      :return: None
      """
      for i in range(len(self.route)):
         randomRate = random.random()
         if randomRate <= mutationRate:
            indexA = random.randint(1, len(self.route) - 1)
            indexB = random.randint(1, len(self.route) - 1)
            self.route[indexA], self.route[indexB] = self.route[indexB], self.route[indexA]

   def __repr__(self):
      return str(self.route)
