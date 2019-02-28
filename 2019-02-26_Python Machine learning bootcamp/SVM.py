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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV


cancer = load_breast_cancer()
df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])

X = df
y = cancer["target"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

paramGrid = {"C": [0.1, 1, 10, 100, 1000], "gamma":[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), paramGrid, verbose=3)
grid.fit(xTrain, yTrain)

print(grid.best_params_)
yPredict = grid.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))