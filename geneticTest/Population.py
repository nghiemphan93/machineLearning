from geneticTest.DNA import DNA
from typing import List
import random

class Population:
   def __init__(self, target: str, mutationRate: float, populationSize: int):
      self.population: List[DNA] = []
      self.matingPool: List[DNA] = []
      self.generations: int = 0
      self.finished: bool = False
      self.target = target
      self.mutationRate = mutationRate
      self.perfectScore = 1.0
      self.best = ""

      for i in range(populationSize):
         self.population.append(DNA(len(self.target)))
      self.calcFitness()

   def calcFitness(self):
      for i in range(len(self.population)):
         self.population[i].calcFitness(self.target)

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
         # fitness = self.population[i].fitness
         n = int(fitness * 100)
         for j in range(n):
            self.matingPool.append(self.population[i])

   def selectAndGenerate(self):
      newPopulation: List[DNA] = []
      for i in range(len(self.population)):
         partnerA = self.pickOne(self.population)
         partnerB = self.pickOne(self.population)
         child = partnerA.crossover(partnerB)
         child.mutate(self.mutationRate)
         newPopulation.append(child)
      self.population = newPopulation
      self.generations += 1

   def generate(self):
      for i in range(len(self.population)):
         indexA = int(random.random() * len(self.matingPool))
         indexB = int(random.random() * len(self.matingPool))
         partnerA = self.matingPool[indexA]
         partnerB = self.matingPool[indexB]
         child: DNA = partnerA.crossover(partnerB)
         child.mutate(self.mutationRate)
         self.population[i] = child
      self.generations += 1

   def getBest(self):
      return self.best

   def pickOne(self, population: List[DNA]):
      index = 0
      rate = random.random()
      total = self.getSumFitness()
      while rate >= 0:
         rate = rate - population[index].fitness / total
         index += 1
      index -= 1
      return population[index]

   def evaluate(self):
      worldRecord = 0.0
      index = 0
      for i in range(len(self.population)):
         if self.population[i].fitness > worldRecord:
            index = i
            worldRecord = self.population[i].fitness
      self.best = self.population[index].getPhrase()
      if worldRecord == self.perfectScore:
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


