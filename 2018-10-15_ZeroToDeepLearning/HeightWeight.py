from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./data/weight-height.csv")

# Add color column
df["GenderColumn"] = df["Gender"].map({"Male": "blue", "Female": "red"})

'''
df.plot(kind="scatter",
        x="Height",
        y="Weight",
        c=df["GenderColumn"],
        alpha=0.4,
        title="male and Female Populations")
plt.show()
'''


males    = df[df["Gender"] == "Male"]
females  = df[df["Gender"] == "Female"]

males["Height"].plot(kind="hist",
                     bins=50,
                     alpha=0.3,
                     color=males["GenderColumn"])
females["Height"].plot(kind="hist",
                     bins=50,
                     alpha=0.3,
                     color=females["GenderColumn"])
plt.axvline(males["Height"].mean(), color="blue", linewidth=2)
plt.axvline(females["Height"].mean(), color="red", linewidth=2)
plt.title("cai lon")

plt.show()
print(females)
