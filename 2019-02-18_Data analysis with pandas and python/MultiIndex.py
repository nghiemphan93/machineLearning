import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/bigmac.csv", parse_dates=["Date"], index_col=["Date", "Country"])
#df = df.set_index(keys=["Date", "Country"])
df = df.sort_index(ascending=[True, False])

print(df)