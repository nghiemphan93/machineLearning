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
      #print("len mating pool: ", len(self.matingPool))
      #self.matingPool.clear()
      self.matingPool = []

      maxFitness = 0
      minFitness = 1
      for i in range(len(self.DNAList)):
         if self.DNAList[i].fitness > maxFitness:
            maxFitness = self.DNAList[i].fitness
         if self.DNAList[i].fitness < minFitness:
            minFitness = self.DNAList[i].fitness
      #print("min fitness: ", minFitness)
      #print("max fitness: ", maxFitness)

      '''
      for i in range(len(self.DNAList)):
         #scaledFitness = map(self.DNAList[i].fitness, 0, maxFitness, 0, 1)
         scaledFitness = self.normalize(self.DNAList[i].fitness, maxFitness, minFitness)
         timesAdded = int(scaledFitness * 100)

         if scaledFitness == 0.0:
            self.matingPool.append(self.DNAList[i])
            x = 0
         else:
            for i in range(timesAdded):
               self.matingPool.append(self.DNAList[i])
      '''
      for i in range(len(self.DNAList)):
         timesAdded = int(self.DNAList[i].fitness * 100)


         for j in range(timesAdded):
            self.matingPool.append(self.DNAList[i])
      #print(len(self.matingPool))

      '''
      totalFitness = self.calcSumFitness()
      if totalFitness == 0:
         for i in range(len(self.DNAList)):
            self.matingPool.append(self.DNAList[i])
      else:
         for i in range(len(self.DNAList)):
            scaledFittness = self.DNAList[i].fitness / totalFitness
            timesAdded = int(scaledFittness*100)
            if timesAdded == 0:
               self.matingPool.append(self.DNAList[i])
            else:
               for i in range(timesAdded):
                  self.matingPool.append(self.DNAList[i])
      '''

   def normalize(self, number, max, min) -> float:
      if (max - min) == 0:
         return 0.0
      else:
         return (number - min) / (max - min)

   def newGeneration(self) -> None:
      for i in range(len(self.DNAList)):
         indexA = int(random.randint(0, len(self.matingPool) - 1))
         indexB = int(random.randint(0, len(self.matingPool) - 1))
         #print("index A: ", indexA)
         #print("len(matingPool): ", len(self.matingPool))

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
      return self.calcSumFitness() / len(self.DNAList)

   def calcSumFitness(self) -> float:
      totalFitness = 0.0
      for i in range(len(self.DNAList)):
         totalFitness += self.DNAList[i].fitness
      return totalFitness

   def getAllPhrases(self) -> str:
      allPhrases = ""
      displayLimit = min(len(self.DNAList), 10)
      for i in range(displayLimit):
         #allPhrases.join(self.DNAList[i].getPhrase() + "\n")
         allPhrases += "{}\n".format(self.DNAList[i].getPhrase())
      return allPhrases