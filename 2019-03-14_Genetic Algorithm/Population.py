import random
from DNA import DNA
from typing import List

class Population:
   def __init__(self, target: str, mutationRate: float, populationSize: int):
      self.DNAList = []
      self.matingPool = []
      self.target       = target
      self.generations = 0
      self.mutationRate = mutationRate
      self.isFinished = False
      self.perfectScore = 1
      for i in range(populationSize):
         newDNA = DNA(target)
         self.DNAList.append(newDNA)
      self.calcFitnessAllMembers()

   def calcFitnessAllMembers(self) -> None:
      for i in range(len(self.DNAList)):
         self.DNAList[i].calcFitness(self.target)

   def naturalSelect(self) -> None:
      self.matingPool.clear()

      maxFitness = 0
      for i in range(len(self.DNAList)):
         if self.DNAList[i].fitness > maxFitness:
            maxFitness = self.DNAList[i].fitness

      for i in range(len(self.DNAList)):
         fitness = map(self.DNAList[i].fitness, 0, maxFitness, 0, 1)
         timesAdded = int(fitness*100)
         for i in range(timesAdded):
            self.matingPool.append(self.DNAList[i])
         print(fitness)

   def newGeneration(self) -> None:
      for i in range(len(self.DNAList)):
         indexA = int(random.randint(len(self.matingPool)))
         indexB = int(random.randint(len(self.matingPool)))
         partnerA: DNA = self.matingPool[indexA]
         partnerB: DNA = self.matingPool[indexB]
         child = partnerA.crossover(partnerB)
         child.mutate(self.mutationRate)
         self.DNAList[i] = child
      self.generations += 1

   def getBestDNA(self) -> DNA:
      bestFitness = 0.0
      bestIndex = 0
      for i in range(len(self.DNAList)):
         if self.DNAList[i].fitness > bestFitness:
            bestIndex = i
            bestFitness = self.DNAList[i].fitness

      if bestFitness == self.perfectScore:
         self.isFinished = True
      return self.DNAList[bestIndex]

   def isFinished(self) -> bool:
      return self.isFinished

   def getGeneration(self) -> int:
      return self.generations

   def calcAverageFitness(self) -> float:
      totalFitness = 0.0
      for i in range(len(self.DNAList)):
         totalFitness += self.DNAList[i].fitness
      return totalFitness / len(self.DNAList)

   def toString(self) -> str:
      allPhrases = ""
      displayLimit = min(len(self.DNAList), 10)
      for i in range(displayLimit):
         allPhrases.join(self.DNAList[i].getPhrase() + "\n")
      return allPhrases