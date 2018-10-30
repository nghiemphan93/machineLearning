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
import requests

'''
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
data = resp.json()

for key, value in data.items():
   print(key, ": ", value)
'''

df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
   'data': range(6)})
dummies = pd.get_dummies(df["key"], prefix="key")
print(dummies)

dfWithDummy = df[["data"]].join(dummies)
print(dfWithDummy)
