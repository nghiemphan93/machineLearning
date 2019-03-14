import random

class DNA:
   def __init__(self, target: str):
      self.genes = []
      self.fitness = 0.0
      for i in range(len(target)):
         self.genes.append(chr(random.randint(32, 128)))

   def getPhrase(self) -> str:
      return "".join(self.genes)

   def calcFitness(self, target: str) -> float:
      score = 0
      for i in range(len(self.genes)):
         if (self.genes[i] == target[i]):
            score += 1
      self.fitness = float(score) / len(target)

   def crossover(self, partner):
      child = DNA(len(self.genes))
      midPoint = random.randint(len(self.genes))

      for i in range(len(self.genes)):
         if i > midPoint:
            child.genes[i] = self.genes[i]
         else:
            child.genes[i] = partner.genes[i]
      return child

   def mutate(self, mutaRate: float) -> None:
      for i in range(len(self.genes)):
         if random.random() < mutaRate:
            self.genes[i] = chr(random.randint(32, 128))

