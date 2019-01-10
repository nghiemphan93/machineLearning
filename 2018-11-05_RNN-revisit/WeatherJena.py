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
import os

folder = "C:/Users/phan/OneDrive - adesso Group/DataSet/weather"
fileName = os.path.join(folder, "jena_climate_2009_2016.csv")

# Preprocess data
print('test')
df = pd.read_csv(fileName)

df["Date Time"] = pd.to_datetime(df["Date Time"])
df["rang"] = pd.Series(range(len(df)), index=df.index)
#df = df.set_index("Date Time")
print(df.index)
print(df.info())
print(df)
df.plot(x="rang", y="T (degC)")
plt.show()

'''
df = df.drop(["Date Time"], axis=1)
data = df.values


# Normalize data
mean = data.mean(axis=0)
data -= mean
std = data.std(axis=0)
data /= std
'''
