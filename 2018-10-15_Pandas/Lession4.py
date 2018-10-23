from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
import quandl
import pickle

def downloadData():
   df = quandl.get("FMAC/HPI", authtoken="79DGCgswis6668sSzLT5")
   pickle_out = open("fiddyStates.pickle", "wb")
   pickle.dump(df, pickle_out)
   pickle_out.close()

def loadData():
   pickleIn = open("fiddyStates.pickle", "rb")
   df = pickle.load(pickleIn)
   return df

def pctChange(df: DataFrame):
   for name in df.columns.values:
      df[name] = (df[name] - df[name][0]) / df[name][0]
   return df


df: DataFrame = pd.read_pickle("fiddyStates.pickle")
#df.rename(columns={"AK": "CLGT"}, inplace=True)

df = pctChange(df)


df.plot()
plt.legend().remove()
plt.show()

