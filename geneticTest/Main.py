from geneticTest.Population import Population

target = "to be or not to be"
mutationRate = 0.01
populationSize = 200
population = Population(target, mutationRate, populationSize)

nthGeneration = 0
while population.isFinished() == False:
   population.naturalSelection()
   population.generate()
   # population.selectAndGenerate()
   population.calcFitness()
   population.evaluate()
   print(nthGeneration, " ", population.getBest())
   # print(len(population.matingPool))
   nthGeneration += 1

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