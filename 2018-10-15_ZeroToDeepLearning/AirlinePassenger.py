from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./data/international-airline-passengers.csv")

print(df.info())
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month")
print(df.info())

diagram = df.plot(x)
diagram.set_xlabel("x")
diagram.set_ylabel("Y")
plt.show()
