from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame


df: DataFrame = pd.read_csv("http://bit.ly/uforeports")

#df = df["Time"].str.slice(-5, -3)
df["Time"] = pd.to_datetime(df["Time"])
df["year"] = df["Time"].dt.year

print(df["year"].value_counts().sort_index())

#df["year"].value_counts().sort_index().plot()

df = df.set_index("Time")


plt.show()