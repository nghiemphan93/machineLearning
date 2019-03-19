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
         if self.genes[i] == target[i]:
            score += 1
      self.fitness = float(score) / len(target)
      #self.fitness = float(score)

   def crossover(self, partner):
      child = DNA("".join(self.genes))
      # print("child 1: ", child.getPhrase())
      midPoint = random.randint(0, len(self.genes) - 1)
      #print(midPoint)

      for i in range(len(self.genes)):
         if i > midPoint:
            child.genes[i] = self.genes[i]
         else:
            child.genes[i] = partner.genes[i]
      # print("child 2: ", child.getPhrase())
      return child

   def mutate(self, mutaRate: float) -> None:
      for i in range(len(self.genes)):
         if random.random() < mutaRate:
            self.genes[i] = chr(random.randint(32, 128))

