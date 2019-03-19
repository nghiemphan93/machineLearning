from typing import List
import random
from random import Random
from genetic.TSP.City import City
import math
import numpy as np


class DNA:
   def __init__(self, route: List[City]):
      self.random = Random()
      self.route: List[City] = route.copy()
      self.route = self.shuffleRoute()
      self.fitness = self.calcFitness()
      self.distance = self.calcTotalDist()

   def shuffleRoute(self):
      newRoute: List[City] = self.route[1:]
      self.random.shuffle(newRoute)
      newRoute.insert(0, self.route[0])
      return newRoute

   def calcFitness(self):
      totalDistance = self.calcTotalDist()
      self.fitness = 1 / totalDistance
      return self.fitness

   def calcTotalDist(self) -> float:
      totalDist = 0.0
      for i in range(len(self.route) - 1):
         totalDist = totalDist + self.route[i].calcDistance(self.route[i + 1])
      totalDist = totalDist + self.route[-1].calcDistance(self.route[0])
      self.distance = totalDist
      return totalDist

   def getRoute(self):
      return self.route

   def __repr__(self):
      return str(self.route)

   # def crossover2(self, partner) -> []:
   #    orderA = self.route
   #    orderB = partner.genes
   #    startIndex = int(random.randint(0, len(orderA) - 2))
   #    endIndex = int(random.randint(startIndex + 1, len(orderA) - 1))
   #    newOrder = orderA[startIndex: endIndex]
   #    for i in range(len(orderB)):
   #       if orderB[i] not in newOrder:
   #          newOrder.append(orderB[i])
   #    child = DNA(route=newOrder)
   #    child.route = newOrder
   #    return newOrder

   # def breed(self, parent1, parent2):
   #    child = parent1
   #    childP1 = []
   #    childP2 = []
   #    geneA = int(random.random() * len(parent1.route))
   #    geneB = int(random.random() * len(parent1.route))
   #    startGene = min(geneA, geneB)
   #    endGene = max(geneA, geneB)
   #    for i in range(startGene, endGene):
   #       childP1.append(parent1.route[i])
   #    childP2 = [item for item in parent2.route if item not in childP1]
   #    child.route = childP1 + childP2
   #    return child

   def crossover(self, partner) -> []:
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

   # def mutate(self, mutationRate: float):
   #    for i in range(1, len(self.route) - 2):
   #       randomRate = random.random()
   #       if randomRate <= mutationRate:
   #          tempCity = self.route[i]
   #          self.route[i] = self.route[i + 1]
   #          self.route[i + 1] = tempCity

   def mutate(self, mutationRate: float):
      for i in range(1, len(self.route) - 2):
         randomRate = random.random()
         if randomRate <= mutationRate:
            tempCity = self.route[i]
            self.route[i] = self.route[i + 1]
            self.route[i + 1] = tempCity