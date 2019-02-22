import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/fortune1000.csv", index_col="Rank")
fortune = pd.DataFrame(columns=df.columns)
sectors = df.groupby(by="Sector")

#print(sectors.agg({"Revenue": "sum","Profits": "sum", "Employees": "mean"}))
#print(sectors.agg(["size", "sum", "mean"]))

for sector, data in sectors:
   fortune = fortune.append(data.nlargest(1, "Revenue"))

cities = df.groupby("Location")
fortune2 = pd.DataFrame(columns=df.columns)

for city, data in cities:
   fortune2 = fortune2.append(data.nlargest(1, "Revenue"))
print(fortune2)