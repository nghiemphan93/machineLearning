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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sklearn.metrics as metrics
# endregion

iris = load_iris()
df: pd.DataFrame = pd.DataFrame(data=iris["data"],
                                columns=iris["feature_names"])
df["target"] = pd.Series(data=iris["target"])
X = df.drop("target", axis=1)
y = df["target"]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.4)

'''
svm = SVC(gamma="scale")
svm.fit(xTrain, yTrain)
yPredict = svm.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
'''

paramGrid = paramGrid = {"C": [0.1, 1, 10, 100, 1000], "gamma":[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid=paramGrid, verbose=3)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
yPredict = grid.predict(xTest)
print(metrics.accuracy_score(yPredict, yTest))
