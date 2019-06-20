from tkinter import *

from geneticTSP.Population import Population


def animation() -> None:
   """
   Redraw the canvas after each generation

   :return: None
   """
   canvas.delete("all")
   population.naturalSelection()
   population.generate()
   population.calcFitness()
   population.calcDistances()
   population.evaluate()

   drawCities(pop=population)
   canvas.after(int(1000 / 60), animation)


def drawCities(pop: Population) -> None:
   """
   Draw all cities and current best route

   :param pop: Population
   :return: None
   """
   global nthGeneration
   r = 10
   for i in range(len(pop.cityList)):
      canvas.create_oval(pop.cityList[i].x - r,
                         pop.cityList[i].y - r,
                         pop.cityList[i].x + r,
                         pop.cityList[i].y + r)
   for i in range(len(pop.bestDNA.route) - 1):
      canvas.create_line(pop.bestDNA.route[i].x,
                         pop.bestDNA.route[i].y,
                         pop.bestDNA.route[i + 1].x,
                         pop.bestDNA.route[i + 1].y)
   canvas.create_line(pop.bestDNA.route[-1].x,
                      pop.bestDNA.route[-1].y,
                      pop.bestDNA.route[0].x,
                      pop.bestDNA.route[0].y)
   print(nthGeneration, " ", pop.bestDNA.distances, " ", pop.bestDNA)
   nthGeneration += 1


if __name__ == "__main__":
   numbCities = 20  # number of cities
   mutationRate = 0.1  # rate of diversity, large means more diverse
   populationSize = 100  # size of each generation
   eliteRate = 0.1  # how many current best DNAs to keep for next generation
   population = Population(numbCities=numbCities,
                           mutationRate=mutationRate,
                           populationSize=populationSize,
                           eliteRate=eliteRate)

   nthGeneration = 0
   root = Tk()
   canvas = Canvas(root, height=750, width=750)
   canvas.pack()
   animation()
   root.mainloop()
