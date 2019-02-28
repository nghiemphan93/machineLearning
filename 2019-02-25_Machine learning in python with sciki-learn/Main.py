import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


iris = load_iris()
data = iris.data
featureNames = iris.feature_names
target = iris.target
targetNames = iris.target_names

X = data
y = target
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=4)

# KNN
kRange = range(1, 26)
scores = []
for k in kRange:
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(xTrain, yTrain)
   yPredict = knn.predict(xTest)
   scores.append(metrics.accuracy_score(y_true=yTest, y_pred=yPredict))
scoreSerie = pd.Series(data=scores, index=range(1, len(scores)+1))
scoreSerie.plot(kind="line")
plt.show()
print(scoreSerie)


# Logistic Regression
logreg = LogisticRegression()
#indices = np.arange(X.shape[0])
#np.random.shuffle(indices)
#X = X[indices]
#y = y[indices]
#yPredict = logreg.predict(XTest)
'''
logreg.fit(xTrain, yTrain)
yPredict = logreg.predict(xTest)
print(metrics.accuracy_score(y_true=yTest, y_pred=yPredict))
'''

