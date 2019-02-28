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
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = pd.DataFrame(iris["target"], columns=["target"])

def convertName(index: int):
   return iris["target_names"][index]
#y["target_names"] = y["target"].apply(convertName)
clgt: pd.DataFrame = pd.concat([X, y], axis=1)
clgt["target_names"] = clgt["target"].apply(convertName)
print(clgt)
#sns.pairplot(clgt, hue="target_names")
#sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", data=clgt, hue="target_names")
#plt.show()

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
'''
svc = SVC()
svc.fit(xTrain, yTrain)
yPredict = svc.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
'''

paramGrid = {"C": [0.1, 1, 10, 100, 1000], "gamma":[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid=paramGrid, verbose=3)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
yPredict = grid.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))

