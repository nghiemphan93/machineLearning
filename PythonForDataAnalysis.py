from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression
import pandas_datareader.data as web

columns = ["AAPL", "IBM", "MSFT", "GOOG"]
df = web.get_data_yahoo(columns)
df = df["Adj Close"]
df["pctChange"] = (df["AAPL"] - df.loc[0, "AAPL"]) / df.iloc[0, 0]

print(df)
