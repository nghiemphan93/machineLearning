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

df = pd.read_csv("./data/advertising.csv")
print(df)

#sns.distplot(df["Age"])
#sns.jointplot(x="Age", y="Area Income", data=df)
#sns.jointplot(x="Age", y="Daily Time Spent on Site", data=df, kind="kde")
#sns.jointplot(x="Daily Time Spent on Site", y="Daily Internet Usage", data=df, kind="kde")
#plt.figure(figsize=(12, 12))
#sns.pairplot(data=df, hue="Clicked on Ad")
#plt.show()

y = df["Clicked on Ad"]
X = df.drop(["Ad Topic Line", "City", "Country", "Timestamp"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
logModel = LogisticRegression()
logModel.fit(X, y)
yPredict = logModel.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))
