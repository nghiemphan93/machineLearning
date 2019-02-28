import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import numpy as np
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("http://bit.ly/kaggletrain")

featureCols = ["Pclass", "Parch"]
X = df.loc[:, featureCols]
y = df["Survived"]

logReg = LogisticRegression()
logReg.fit(X, y)

test = pd.read_csv("http://bit.ly/kaggletest")
xTest = test.loc[:, featureCols]

yPredict = logReg.predict(xTest)

result = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": yPredict})
result = result.sort_values("PassengerId", axis="rows")
#result = result.set_index(keys="PassengerId")

print(result)