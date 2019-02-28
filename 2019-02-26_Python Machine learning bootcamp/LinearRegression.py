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

df = pd.read_csv("./data/USA_Housing.csv")
print(df)
#sns.pairplot(df)
#sns.distplot(df["Price"])
#sns.heatmap(df.corr())
#plt.show()

X = df.drop(["Price", "Address"], axis=1)
y = df["Price"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.4)

linReg = LinearRegression()
linReg.fit(xTrain, yTrain)
yPredict = linReg.predict(xTest)

print(np.sqrt(metrics.mean_squared_error(y_true=yTest, y_pred=yPredict)))
print(np.mean(np.abs(yPredict - yTest)))

sns.distplot((yTest-yPredict))
plt.show()