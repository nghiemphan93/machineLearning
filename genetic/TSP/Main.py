import random
from random import Random
from genetic.TSP.City import City
from genetic.TSP.DNA import DNA
from genetic.TSP.Helper import Helper
from genetic.TSP.Population import Population

'''
target = "to be or not to be"
mutationRate = 0.01
populationSize = 200
#population = Population(target, mutationRate, populationSize)

counter = 0
while population.isFinished() == False:
   population.naturalSelection()
   population.generate()
   # population.selectAndGenerate()
   population.calcFitness()
   population.evaluate()
   print(counter, " ", population.getBest())
   # print(len(population.matingPool))
   counter += 1
'''
'''
while True:
   dna1: DNA = DNA(len(target))
   dna2: DNA = DNA(len(target))
   child: DNA = dna1.crossover(dna2)
   dna1.calcFitness(target)
   dna2.calcFitness(target)
   child.calcFitness(target)

   print(dna1.getPrase())
   print(dna2.getPrase())
   print(child.getPrase())
   child.mutate(mutationRate)
   print(child.getPrase())

   print(dna1.fitness)
   print(dna2.fitness)
   print(child.fitness)
'''
'''
numbCities = 20
mutationRate = 0.2
populationSize = 200
population = Population(numbCities, mutationRate, populationSize)

cityA = City(x=15, y=20)
cityB = City(x=45, y=100)
cityC = City(x=50, y=200)
cityD = City(x=24, y=70)
cityE = City(x=204, y=140)
cityF = City(x=98, y=340)
cityList = [cityA, cityB, cityC, cityD, cityE, cityF, cityA]
dnaA = DNA(cityList)
dnaB = DNA(cityList)
'''
# for pop in population.population:
#    print(pop.calcTotalDist())

'''
counter = 0
while population.isFinished() == False:
   population.naturalSelection()
   # population.generateOnlyMutate()
   population.generate()
   # population.selectAndGenerate()
   population.calcFitness()
   print(counter, " ", population.getBestDistance())
   counter += 1
'''

# route = [City(432, 197), City(388, 455), City(215, 20), City(132, 494), City(261, 248)]
numbCities = 10
mutationRate = 0.05
populationSize = 200
eliteRate = 0.1
population = Population(numbCities, mutationRate, populationSize, eliteRate)

# route = Helper.createRoute(10)
# 658.7794816134837
route = [City(199, 263), City(225, 398), City(218, 172), City(127, 242), City(318, 286)]
route2 = [City(199, 263), City(225, 398), City(127, 242), City(218, 172), City(318, 286)]
route3 = [City(199, 263), City(218, 172), City(225, 398), City(127, 242), City(318, 286)]
route4 = [City(199, 263), City(218, 172), City(225, 398), City(127, 242), City(318, 286),
          City(432, 197), City(388, 455), City(215, 20), City(132, 494), City(261, 248)]
dna = DNA(route)
dna2 = DNA(route)

dna.route = route3

# population.cityList = route4
# population.population = population.createPopulation(10)
# population.calcFitness()
# population.calcDistance()


counter = 0
while counter <= 20:
   population.naturalSelection()
   population.generate()
   population.calcFitness()
   population.calcDistance()
   population.evaluate()
   print(counter, " ", population.bestDNA.distance, " ", population.bestDNA)
   counter += 1

# print(route)
# print(dna)
# print(dna2)
# child = dna.crossover(dna2)
# print(child)
# child.mutate(mutationRate)
# print(child)
