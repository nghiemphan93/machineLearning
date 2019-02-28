import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer, StandardScaler

df = pd.read_csv("./data/kyphosis.csv")
print(df)

#sns.pairplot(df, hue="Kyphosis")
#plt.show()

X = df.drop("Kyphosis", axis=1)
y = df["Kyphosis"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
'''
dTree = DecisionTreeClassifier()
dTree.fit(X, y)
yPredict = dTree.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
'''

randomForest = RandomForestClassifier(n_estimators=200)
randomForest.fit(xTrain, yTrain)
yPredict = randomForest.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))

print(df["Kyphosis"].value_counts())