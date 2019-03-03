# region Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
# endregion

iris = load_iris()
df: pd.DataFrame = pd.DataFrame(data=iris["data"],
                                columns=iris["feature_names"])
df["target"] = pd.Series(data=iris["target"])
print(df)
print(df["target"].value_counts())

#sns.pairplot(data=df, hue="target")
#sns.countplot(x=df["petal length (cm)"], hue=df["target"])
#plt.show()

X = df.drop("target", axis=1)
y = df["target"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

'''
logReg = LogisticRegression()
logReg.fit(xTrain, yTrain)
yPredict = logReg.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
'''

knn = KNeighborsClassifier()
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))