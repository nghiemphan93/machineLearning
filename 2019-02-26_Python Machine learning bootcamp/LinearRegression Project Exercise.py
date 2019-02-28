import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.datasets import load_boston

df = pd.read_csv("./data/Ecommerce Customers")
print(df)

#sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df)
#sns.pairplot(data=df)
#sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df)
#plt.show()

X = df.drop(["Email", "Address", "Avatar", "Yearly Amount Spent"], axis=1)
y = df["Yearly Amount Spent"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

linReg = LinearRegression()
linReg.fit(xTrain, yTrain)
yPredict = linReg.predict(xTest)

#plt.scatter(x=yPredict, y=yTest)
sns.distplot((yPredict-yTest))
plt.show()

print(metrics.mean_absolute_error(yPredict, yTest))
print(linReg.coef_)