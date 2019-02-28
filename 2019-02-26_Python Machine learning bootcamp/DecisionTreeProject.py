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

df = pd.read_csv("./data/loan_data.csv")


'''
df[df["credit.policy"] == 1]["fico"].plot.hist(color="blue", alpha=0.5, label="Credit Policy 1", bins=30)
df[df["credit.policy"] == 0]["fico"].plot.hist(color="red", alpha=0.5, label="Credit Policy 2", bins=30)
plt.legend()
plt.show()
'''
'''
df[df["not.fully.paid"] == 1]["fico"].plot.hist(color="blue", alpha=0.5, label="Credit Policy 1", bins=30)
df[df["not.fully.paid"] == 0]["fico"].plot.hist(color="red", alpha=0.5, label="Credit Policy 2", bins=30)
plt.legend()
plt.show()
'''
'''
plt.figure(figsize=(12, 4))
sns.countplot(x="purpose", data=df, hue="not.fully.paid")
plt.show()
'''
'''
#sns.jointplot(x="fico", y="int.rate", data=df, kind="kde")
plt.figure(figsize=(12, 4))
sns.lmplot(x="fico", y="int.rate", data=df, hue="credit.policy", col="not.fully.paid")
plt.show()
'''

purposeDummies = pd.get_dummies(df["purpose"], drop_first=True)
df = pd.concat([df, purposeDummies], axis=1)

X = df.drop(["purpose", "not.fully.paid"], axis=1)
y = df["not.fully.paid"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
'''
dTree = DecisionTreeClassifier()
dTree.fit(xTrain, yTrain)
yPredict = dTree.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
'''

randomForest = RandomForestClassifier(n_estimators=600)
randomForest.fit(xTrain, yTrain)
yPredict = randomForest.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))