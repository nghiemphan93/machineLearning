import random
from DNA import DNA
from Population import Population




population = Population(target="cai lon",
                        mutationRate=0.01 ,
                        populationSize=10)
print(population.calcAverageFitness())

'''
target = "cai lon"
for i in range(3):
   print(DNA(target).getPhrase())
'''