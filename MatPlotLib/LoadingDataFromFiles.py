import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

with open("example", "r") as csvfile:
    plots = csv.reader(csvfile, delimiter=",")
    for row in plots:
        print(row)
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x, y, label="Loaded from file")

plt.xlabel("Plot Number")
plt.ylabel("Important var")
plt.title("Interesting Graph\nCheck it out")
plt.legend()

plt.show()
