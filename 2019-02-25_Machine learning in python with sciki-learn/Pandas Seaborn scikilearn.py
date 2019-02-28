import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")

X = df[["TV", "radio", "newspaper"]].values
y = df["sales"].values

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
linreg = LinearRegression()
linreg.fit(xTrain, yTrain)

yPredict = linreg.predict(xTest)
score = metrics.mean_absolute_error(y_true=yTest, y_pred=yPredict)
print(score)