from typing import List
import random


class DNA:
   def __init__(self, targetLength: int):
      self.genes: chr = []
      self.fitness = 0
      for i in range(targetLength):
         # self.genes.append(self.newChar())
         self.genes.append(self.newCharNormal())

   def newChar(self) -> chr:
      c = int(random.random() * 122 + 63)
      if c == 63:
         c = 32
      if c == 64:
         c = 46
      return chr(c)

   def newCharNormal(self) -> chr:
      c = int(random.random() * 128 + 32)
      return chr(c)

   def getPhrase(self):
      return "".join(self.genes)

   def calcFitness(self, target: str):
      score = 0.0
      for i in range(len(self.genes)):
         if self.genes[i] == target[i]:
            score = score + 1
      self.fitness = score / len(target)

   def crossover(self, partner):
      child = DNA(len(self.genes))
      midpoint = int(random.random() * len(self.genes))
      for i in range(len(self.genes)):
         if i > midpoint:
            child.genes[i] = self.genes[i]
         else:
            child.genes[i] = partner.genes[i]
      return child

   def mutate(self, mutationRate: float):
      for i in range(len(self.genes)):
         randomRate = random.random()
         if randomRate < mutationRate:
            # self.genes[i] = self.newChar()
            self.genes[i] = self.newCharNormal()