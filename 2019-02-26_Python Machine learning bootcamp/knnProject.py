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

df = pd.read_csv("./data/KNN_Project_Data")

#sns.pairplot(df, hue="TARGET CLASS")
#plt.show()

scaler = StandardScaler()
scaler.fit(df.drop("TARGET CLASS", axis=1))
scaledFeatures = scaler.transform(df.drop("TARGET CLASS", axis=1))
dfScaled = pd.DataFrame(scaledFeatures, columns=df.columns[:-1])

print(dfScaled)

X = dfScaled
y = df["TARGET CLASS"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
accuracy = []
for k in range(1, 40):
   knn = KNeighborsClassifier(k)
   knn.fit(X, y)
   yPredict = knn.predict(xTest)
   accuracy.append(metrics.accuracy_score(yPredict, yTest))
kNumber = [i for i in range(1, 40)]
accDF = pd.DataFrame({"K Neighbors": kNumber, "Accuracy": accuracy})
print(accDF)

sns.scatterplot(x="K Neighbors", y="Accuracy", data=accDF)
plt.show()
