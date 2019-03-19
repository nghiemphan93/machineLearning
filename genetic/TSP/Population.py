from genetic.TSP.DNA import DNA
from genetic.TSP.City import City
from typing import List
import random

class Population:
   def __init__(self, numbCities: int, mutationRate: float, populationSize: int, eliteRate: float):
      self.eliteRate = eliteRate
      self.matingPool: List[DNA] = []
      self.generations: int = 0
      self.finished: bool = False
      self.numbCities = numbCities
      self.mutationRate = mutationRate
      self.perfectScore = 1.0
      self.bestDNA: DNA = None
      self.cityList: List[City] = self.createRoute(self.numbCities)
      self.population: List[DNA] = self.createPopulation(populationSize)
      self.calcFitness()

   def createRoute(self, numbCities: int) -> List[City]:
      # cityRoute = []
      # for i in range(numbCities):
      #    cityRoute.append(City(x=random.randint(0, 500),
      #                          y=random.randint(0, 500)))
      cityRoute = [City(199, 263), City(218, 172), City(225, 398), City(127, 242), City(318, 286),
          City(432, 197), City(388, 455), City(215, 20), City(132, 494), City(261, 248)]
      return cityRoute

   def createPopulation(self, populationSize) -> List[DNA]:
      population = []
      for i in range(populationSize):
         population.append(DNA(route=self.cityList))
      return population

   def calcFitness(self):
      for i in range(len(self.population)):
         self.population[i].calcFitness()

   def calcDistance(self):
      for i in range(len(self.population)):
         self.population[i].calcTotalDist()

   def naturalSelection(self):
      self.matingPool = []

      maxFitness = 0
      minFitness = 1
      for i in range(len(self.population)):
         if self.population[i].fitness > maxFitness:
            maxFitness = self.population[i].fitness
         if self.population[i].fitness < minFitness:
            minFitness = self.population[i].fitness

      for i in range(len(self.population)):
         fitness = (self.population[i].fitness - minFitness) / (maxFitness - minFitness)
         n = int(fitness * 1000)
         for j in range(n):
            self.matingPool.append(self.population[i])

   def rankPopulation(self):
      self.calcDistance()
      self.population.sort(key=lambda route: route.distance)

   def generate(self):
      self.rankPopulation()
      eliteSize = int(len(self.population) * self.eliteRate)
      for i in range(eliteSize, len(self.population)):
         indexA = int(random.random() * len(self.matingPool))
         indexB = int(random.random() * len(self.matingPool))
         partnerA = self.matingPool[indexA]
         partnerB = self.matingPool[indexB]
         child: DNA = partnerA.crossover(partnerB)
         child.mutate(self.mutationRate)
         self.population[i] = child
      self.generations += 1

   def pickOne(self, population: List[DNA]):
      index = 0
      rate = random.random()
      total = self.getSumFitness()
      while rate >= 0:
         rate = rate - population[index].fitness / total
         index += 1
      index -= 1
      return population[index]

   def getBestFitness(self):
      bestFitness = 0.0
      for i in range(len(self.population)):
         if self.population[i].fitness > bestFitness:
            bestFitness = self.population[i].fitness
      return bestFitness

   def getBestDistance(self):
      bestDistance = 999999
      for i in range(len(self.population)):
         totalDist = self.population[i].calcTotalDist()

         if totalDist < bestDistance:
            bestDistance = totalDist
      return bestDistance

   def evaluate(self):
      bestFitness = 0.0
      indexBestDNA = 0
      for i in range(len(self.population)):
         if self.population[i].fitness > bestFitness:
            indexBestDNA = i
            bestFitness = self.population[i].fitness
      self.bestDNA = self.population[indexBestDNA]
      if bestFitness == self.perfectScore:
         self.finished = True

   def isFinished(self):
      return self.finished

   def getGenerations(self):
      return self.generations

   def getSumFitness(self):
      total = 0.0
      for i in range(len(self.population)):
         total = total + self.population[i].fitness
      return total

   def getAverageFitness(self):
      return self.getSumFitness() / len(self.population)
