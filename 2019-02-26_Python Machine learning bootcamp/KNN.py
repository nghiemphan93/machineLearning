import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer, StandardScaler

df: pd.DataFrame = pd.read_csv("./data/Classified Data", index_col=0)
scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis=1))
scaledFeatures = scaler.transform(df.drop("TARGET CLASS", axis=1))

dfScaled = pd.DataFrame(scaledFeatures, columns=df.columns[:-1])
X = dfScaled
y = df["TARGET CLASS"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(17)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))



x = 10
print(x)












