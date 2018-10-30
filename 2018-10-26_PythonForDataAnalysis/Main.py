from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame
import pandas_datareader.data as web


df = web.get_data_yahoo(["AAPL", "GOOG", "IBM", "MSFT"])
df = df["Adj Close"]
columns = ['AAPL', 'GOOG', 'IBM', 'MSFT']

for column in columns:
   print(column)

print(df)
print("Percentage Change")
print(df.pct_change())