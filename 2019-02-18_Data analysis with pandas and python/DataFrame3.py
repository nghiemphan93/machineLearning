import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/jamesbond.csv", index_col="Film")
df = df.sort_index()
#print(df.loc[["Moonraker", "Octopussy"]])
df.loc[df["Actor"] == "Sean Connery", "Actor"] = "clgt"
print(df)

