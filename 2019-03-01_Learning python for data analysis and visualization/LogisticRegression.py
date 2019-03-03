# region Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import statsmodels.api as sm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
# endregion

df: pd.DataFrame = sm.datasets.fair.load_pandas().data
def hasAffair(amount):
   if amount > 0:
      return 1
   else:
      return 0
df["has affair"] = df["affairs"].apply(hasAffair)

#sns.countplot(x="has affair", data=df)
#sns.countplot(x="age", data=df, hue="has affair")
#sns.countplot("yrs_married", data=df, hue="has affair")
#sns.countplot("children", data=df, hue="has affair")
#sns.countplot("educ", data=df, hue="has affair")
#plt.show()

occDummies = pd.get_dummies(data=df["occupation"])
occDummies.columns = ["occ1", "occ2", "occ3", "occ4", "occ5", "occ6"]
occHusbDummies = pd.get_dummies(data=df["occupation_husb"])
occHusbDummies.columns = ["hocc1", "hocc2", "hocc3", "hocc4", "hocc5", "hocc6"]
df = pd.concat(objs=[df, occDummies, occHusbDummies], axis=1, )
df = df.drop(["occupation", "occupation_husb", "affairs", "occ1", "hocc1"], axis=1)


hasAffairDF = df[df["has affair"] == 1]
hasNoAffairDF = df[df["has affair"] == 0].head(2053)

X: pd.DataFrame = pd.concat([hasAffairDF.drop("has affair", axis=1),
                             hasNoAffairDF.drop("has affair", axis=1)],
                            axis=0)
y = pd.DataFrame = pd.concat([hasAffairDF["has affair"],
                             hasNoAffairDF["has affair"]],
                            axis=0)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
logReg = LogisticRegression()
logReg.fit(xTrain, yTrain)
yPredict = logReg.predict(xTest)

print(metrics.accuracy_score(yPredict, yTest))
