# region import
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
import math, datetime
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
# endregion

df = quandl.get("WIKI/GOOGL", authtoken="79DGCgswis6668sSzLT5")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]
forecastCol = "Adj. Close"
df.fillna(-99999, inplace=True)
forecastOut = int(math.ceil(0.01 * len(df)))
df["label"] = df[forecastCol].shift(-forecastOut)
original = df.copy()
df.dropna(inplace=True)

X = np.array(df.drop(["label"], axis=1))
y = np.array(df["label"])

X = preprocessing.scale(X)


xTrain, xTest, yTrain, yTest  = train_test_split(X, y, test_size=0.2)


def training():
   classifier = LinearRegression()
   classifier.fit(xTrain, yTrain)

   with open("linearRegression.pickle", "wb") as file:
      pickle.dump(classifier, file)


def loadModel():
   pickleIn = open("linearRegression.pickle", "rb")
   classifier = pickle.load(pickleIn)
   accuracy = classifier.score(xTest, yTest)
   predicted = classifier.predict(xTest)
   print(accuracy)

loadModel()

'''
original["forecast"] = np.nan
lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in predicted:
   nextDate = datetime.datetime.fromtimestamp(nextUnix)
   nextUnix = nextUnix + oneDay
   original.loc[nextDate, "forecast"] = i

print(original.tail(300))
'''


'''
print(df)
print(df.info())
plt.show()
'''