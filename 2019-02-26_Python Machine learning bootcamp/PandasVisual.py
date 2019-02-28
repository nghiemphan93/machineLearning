import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)

df: pd.DataFrame = pd.read_csv("df3")

#df.plot.scatter(x="a", y="b", figsize=(12, 3), s=50)
#df["a"].plot.hist()
#df[["a", "b"]].plot.box()
df["a"].plot.kde()

print(df)
plt.show()