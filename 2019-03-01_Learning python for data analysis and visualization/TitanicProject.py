# region Import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = pd.read_csv("./data/titanic_train.csv")


'''
#sns.countplot(x="Sex", data=df, hue="Pclass")
sns.countplot(x="Pclass", data=df, hue="Sex")
plt.show()
'''

def maleFemaleChild(passenger):
   age, sex = passenger
   if age < 16:
      return "child"
   else:
      return sex

df["person"] = df[["Age", "Sex"]].apply(maleFemaleChild, axis=1)

#sns.countplot(x="Pclass", data=df, hue="person")
#sns.distplot(df["Age"])
#df["Age"].plot.hist(bins=30)
#print(df["person"].value_counts())
'''
fig = sns.FacetGrid(data=df, hue="Sex", aspect=3)
fig.map(sns.kdeplot, "Age", shade=True)
oldest = df["Age"].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()
'''
'''
fig = sns.FacetGrid(data=df, hue="person", aspect=3)
fig.map(sns.distplot, "Age")
oldest = df["Age"].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()
'''
'''
fig = sns.FacetGrid(data=df, hue="Pclass", aspect=3)
fig.map(sns.kdeplot, "Age", shade=True)
oldest = df["Age"].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
plt.show()
'''

def cabinToDeck(cabin):
   if cabin is not np.nan:
      return cabin[0]
   else:
      return cabin
df["deck"] = df["Cabin"].apply(cabinToDeck)
'''
sns.countplot(x="deck", data=df.dropna().sort_values(by="deck"))
plt.show()
'''
'''
sns.countplot(x="Embarked", data=df, hue="Pclass")
plt.show()
'''

def isAlone(info):
   sibling, parentOrChild = info
   if sibling == 0 and parentOrChild == 0:
      return 1
   else:
      return 0
df["alone"] = df[["SibSp", "Parch"]].apply(isAlone, axis=1)
print(df)
'''
sns.countplot(df["alone"])
plt.show()
'''
#sns.catplot(x="Pclass", y="Survived", data=df)
#sns.lmplot(x="Age", y="Survived", data=df)
#sns.lmplot(x="Age", y="Survived", data=df, col="Pclass")

plt.show()
