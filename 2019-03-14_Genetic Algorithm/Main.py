import random
from DNA import DNA
from Population import Population

# Setup
target = "unicorn"
populationSize = 200
mutationRate = 0.1
population = Population(target=target,
                        mutationRate=mutationRate,
                        populationSize=populationSize)

# Run genetic algorithm
while population.isFinished == False:
   population.naturalSelect()
   population.newGeneration()
   population.calcFitnessAllMembers()
   print(population.getBestDNA().getPhrase(), " ", population.calcAverageFitness())
   #print(len(population.matingPool))

'''
characters = []
for i in range(32, 129):
   characters.append(chr(i))
print(characters)
'''