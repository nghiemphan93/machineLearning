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
from sklearn import preprocessing, svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
# endregion

columns = ["id", "clumpThickness", "uniformCellSize", "uniformCellShape", "marginalAdhesion", "singleCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "Mitoses", "class"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=columns)
df.replace("?", -99999, inplace=True)

'''
for i in range(len(df)):
   for j in range(len(df.columns)):
      if df.iloc[i, j] == -99999:
         print(df.iloc[i])
print(df[df["bareNuclei"] == -99999])
'''
df.drop(["id"], axis=1, inplace=True)

X = np.array(df.drop(["class"], axis=1))
y = np.array(df["class"])

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
classifier = neighbors.KNeighborsClassifier()
classifier.fit(xTrain, yTrain)

accu = classifier.score(xTest, yTest)
print(accu)

example = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example - example.reshape(len(example), -1)

print(example.shape)

prediction = classifier.predict(example)
print(prediction)

