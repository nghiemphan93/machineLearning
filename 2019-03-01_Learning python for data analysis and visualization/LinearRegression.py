# region Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
# endregion

boston = load_boston()


df = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
df["price"] = pd.Series(data=boston["target"])

X = df.drop(["price"], axis=1)
y = df["price"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

ligReg = LinearRegression()
ligReg.fit(xTrain, yTrain)
yPredict = ligReg.predict(xTest)
print(metrics.mean_absolute_error(yPredict, yTest))
coeff = pd.DataFrame(data={"features": df.columns[:-1],
                           "coefficient": ligReg.coef_})
print(coeff)