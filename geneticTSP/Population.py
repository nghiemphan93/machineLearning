from geneticTSP.DNA import DNA
from geneticTSP.City import City
from typing import List
import random


class Population:
   """
   Representation of each generation holding n individual DNAs
   """

   def __init__(self, numbCities: int, mutationRate: float, populationSize: int, eliteRate: float):
      self.eliteRate = eliteRate
      self.rouletteWheel: List[DNA] = []
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
      """
      Randomly create a demo list of cities with x, y coordinates range from 50 to 700

      :param numbCities:
      :return:
      """
      cityRoute = []
      for i in range(numbCities):
         cityRoute.append(City(x=random.randint(50, 700),
                               y=random.randint(50, 700)))
      return cityRoute

   def createPopulation(self, populationSize: int) -> List[DNA]:
      """
      Create all individual DNAs for a generation

      :param populationSize: int
      :return: population: List[DNA]
      """
      population = []
      for i in range(populationSize):
         population.append(DNA(route=self.cityList))
      return population

   def calcFitness(self) -> None:
      """
      Calculate fitness of each DNA inversely proportional to total distances

      :return: None
      """
      for i in range(len(self.population)):
         self.population[i].calcFitness()

   def calcDistances(self) -> None:
      """
      Calculate distances of every DNA

      :return: None
      """
      for i in range(len(self.population)):
         self.population[i].calcTotalDist()

   def naturalSelection(self) -> None:
      """
      Simulate a roulette wheel of all possible DNA candidates to do cross over for next generation

      :return: None
      """
      self.rouletteWheel = []

      maxFitness = 0
      minFitness = 1

      for dna in self.population:
         if dna.fitness > maxFitness:
            maxFitness = dna.fitness
         if dna.fitness < minFitness:
            minFitness = dna.fitness

      for dna in self.population:
         # scale fitness down to 0-1
         fitness = (dna.fitness - minFitness) / (maxFitness - minFitness)
         n = int(fitness * 100)
         for i in range(n):
            self.rouletteWheel.append(dna)

   def rankPopulation(self) -> None:
      """
      Sort the population ascending by distance

      :return: None
      """
      self.calcDistances()
      self.population.sort(key=lambda route: route.distances)

   def generate(self) -> None:
      """
      Pick parents from roulette wheel to create new generation

      :return: None
      """
      self.rankPopulation()
      eliteSize = int(len(self.population) * self.eliteRate)
      for i in range(eliteSize, len(self.population)):
         indexA = int(random.random() * len(self.rouletteWheel))
         indexB = int(random.random() * len(self.rouletteWheel))
         partnerA = self.rouletteWheel[indexA]
         partnerB = self.rouletteWheel[indexB]
         child: DNA = partnerA.crossover(partnerB)
         child.mutate(self.mutationRate)
         self.population[i] = child
      self.generations += 1

   def getBestFitness(self) -> float:
      """
      Get fitness of the best fitted DNA

      :return: bestFitness: float
      """
      bestFitness = 0.0
      for i in range(len(self.population)):
         if self.population[i].fitness > bestFitness:
            bestFitness = self.population[i].fitness
      return bestFitness

   def getBestDistance(self) -> float:
      """
      Get distance of the best fitted DNA

      :return:bestDistance: float
      """
      bestDistance = 999999
      for i in range(len(self.population)):
         totalDist = self.population[i].calcTotalDist()

         if totalDist < bestDistance:
            bestDistance = totalDist
      return bestDistance

   def evaluate(self) -> None:
      """
      Evaluate the process if best solution found

      :return: None
      """
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
