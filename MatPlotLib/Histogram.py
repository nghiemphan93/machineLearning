import matplotlib.pyplot as plt

populationAges = [22,55,62,45,21,22,34,42,42,3,4,99,43,76,110,120,122,130,111,115,112,80,75,65,54,44,43,42,48]
#ids = [x for x in range(len(populationAges))]

bins = [0,10,20,30,40,50,60,70,80,90,100,110, 120,130]

plt.hist(populationAges, bins, histtype="bar", rwidth=0.8)

#plt.bar(ids, populationAges, label="clgt")


plt.xlabel("Plot Number")
plt.ylabel("Important var")
plt.title("Interesting Graph\nCheck it out")
plt.legend()

plt.show()
