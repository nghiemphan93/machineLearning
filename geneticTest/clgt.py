import time
import numpy as np
import math

start = time.time()
for i in range(1000000):
   math.sqrt(200 ** 2 + 400 ** 2)

end = time.time()
print(end - start)


start = time.time()
for i in range(1000000):
   np.sqrt(200 ** 2 + 400 ** 2)

end = time.time()
print(end - start)

# def mutate(self, mutationRate: float):
#    for i in range(len(self.route)):
#       randomRate = random.random()
#       if randomRate <= mutationRate:
#          indexA = random.randint(1, len(self.route) - 1)
#          indexB = indexA + 1 if indexA != len(self.route) - 1 else 0
#          self.route[indexA], self.route[indexB] = self.route[indexB], self.route[indexA]


# def createChildNearestNeighbor(self):
#    bestDNA = self.bestDNA
#
#    for i in range(len(bestDNA.route) - 1):
#       if i + 3 <= len(bestDNA.route):
#          nearestNeighborIndex = i + 1
#          for j in range(i + 2, len(bestDNA.route[i + 2:])):
#             if bestDNA.route[i].distanceTo(nextCity=bestDNA.route[j]) < bestDNA.route[i].distanceTo(
#                  nextCity=bestDNA.route[nearestNeighborIndex]):
#                nearestNeighborIndex = j
#          if i + 1 != nearestNeighborIndex:
#             bestDNA.route[i + 1], bestDNA.route[nearestNeighborIndex] = bestDNA.route[nearestNeighborIndex], \
#                                                                         bestDNA.route[i + 1]
#          # print(f'i = {i}, i+1={i+1}, nearest = {nearestNeighborIndex}')
#          # print(bestDNA.route[i].calcDistance(nextCity=bestDNA.route[i+1]), bestDNA.route[i].calcDistance(nextCity=bestDNA.route[nearestNeighborIndex]))
#       else:
#          nearestNeighborIndex = i + 1
#          if bestDNA.route[i].distanceTo(nextCity=bestDNA.route[0]) < bestDNA.route[i].distanceTo(
#               nextCity=bestDNA.route[nearestNeighborIndex]):
#             nearestNeighborIndex = 0
#             bestDNA.route[i + 1], bestDNA.route[nearestNeighborIndex] = bestDNA.route[nearestNeighborIndex], \
#                                                                         bestDNA.route[i + 1]
#          # print(f'i = {i}, i+1={i + 1}, nearest = {nearestNeighborIndex}')
#
#    bestDNA.calcFitness()
#    bestDNA.calcTotalDist()
#    self.population[-1] = bestDNA
#
#    self.rankPopulation()
#    self.evaluate()
#    # self.bestDNA = bestDNA