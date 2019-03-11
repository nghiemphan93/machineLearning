# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 8, 4
#sns.set()
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
# endregion

'''
battles = pd.read_csv("./data/battles.csv")
characterDeath = pd.read_csv("./data/character-deaths.csv")
characterPredictions = pd.read_csv("./data/character-predictions.csv")

print(battles)
print(battles.info())

#sns.barplot(x="year", y="major_death", data=battles)
sns.countplot(x="year", data=battles, hue="major_death")
plt.show()
'''

df = pd.read_csv("./data/Social_Network_Ads.csv")
print(df)
print(df["Purchased"].value_counts())

X = df[["Age", "EstimatedSalary"]].astype(float)
y = df["Purchased"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)
standardScaler = StandardScaler()
standardScaler.fit(X)
xTrain = standardScaler.transform(xTrain)
xTest = standardScaler.transform(xTest)

classifier = GaussianNB()
classifier.fit(xTrain, yTrain)
yPredict = classifier.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))