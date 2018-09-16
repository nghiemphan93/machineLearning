import matplotlib.pyplot as plt

days =[1,2,3,4,5]

sleeping =  [7,8,6,11,7]
eating =    [2,3,4,3,2]
working =   [7,8,7,2,2]
playing =   [8,5,7,8,13]

slices = [7,2,2,13]
activities = ["sleeping", "eating", "working", "playing"]
colors = ["c", "m", "r", "g"]

plt.pie(slices,
        labels=activities,
        colors=colors,
        startangle=90,
        explode=(0,0.2,0,0),
        autopct="%1.2f%%")

plt.xlabel("Plot Number")
plt.ylabel("Important var")
plt.title("Interesting Graph\nCheck it out")
plt.legend()

plt.show()
