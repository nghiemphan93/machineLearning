from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

def lineChart():
   years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
   gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

   plt.plot(years, gdp, c="red")
   plt.title("nominal GDP")
   plt.xlabel("Year")
   plt.ylabel("Billions of $")
   plt.show()

def barChart():
   movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
   numbOscars = [5, 11, 3, 8, 10]

   xs = [_ for _ in range(len(movies))]

   plt.bar(xs, numbOscars)
   plt.ylabel("# of Academy Awards")
   plt.title("My favorite movies")

   plt.xticks(xs, movies)
   plt.show()

def histogram():
   grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
   decile = lambda grade: grade // 10 * 10
   histogram = Counter(decile(grade) for grade in grades)

   print(histogram)


histogram()
