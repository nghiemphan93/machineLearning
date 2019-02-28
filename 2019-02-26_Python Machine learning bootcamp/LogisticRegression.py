import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.datasets import load_boston

train: pd.DataFrame = pd.read_csv("./data/titanic_train.csv")
#sns.heatmap(train.isnull())
#sns.countplot(x="Survived", hue="Sex", data=train)
#sns.countplot(x="Survived", hue="Pclass", data=train)
#sns.distplot(train["Age"].dropna())
#train["Age"].plot.hist()
#sns.countplot(x="SibSp", data=train)
#sns.distplot(train["Fare"])
#plt.figure(figsize=(10, 7))
#sns.boxplot(x="Pclass", y="Age", data=train)
#plt.show()

def imputeAge(cols):
   Age = cols[0]
   Pclass = cols[1]
   if pd.isnull(Age):
      if Pclass == 1:
         return 37
      elif Pclass == 2:
         return 29
      else:
         return 24
   else:
      return Age
train["Age"] = train[["Age", "Pclass"]].apply(imputeAge, axis=1)
train = train.drop("Cabin", axis=1)

#sns.heatmap(train.isnull())
#sns.boxplot(x="Pclass", y="Age", data=train)
#plt.show()

sex = pd.get_dummies(train["Sex"], drop_first=True)
embarked = pd.get_dummies(train["Embarked"], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1)
train = train.drop(["PassengerId", "Name", "Sex", "Ticket", "Embarked"], axis=1)

X = train.drop("Survived", axis=1)
y = train["Survived"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)
logmodel = LogisticRegression()
logmodel.fit(xTrain, yTrain)
yPredict = logmodel.predict(xTest)

print(metrics.classification_report(yTest, yPredict))

print(metrics.accuracy_score(yPredict, yTest))