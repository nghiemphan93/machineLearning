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

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

df = pd.read_csv("./data/College_Data", index_col=0)

'''
#sns.scatterplot(x="Room.Board", y="Grad.Rate", data=df, hue="Private")
#sns.lmplot(x="Room.Board", y="Grad.Rate", data=df, hue="Private", fit_reg=False)
#sns.lmplot(x="Outstate", y="F.Undergrad", data=df, hue="Private", fit_reg=False)
df[df["Private"] == "Yes"]["Outstate"].plot.hist(alpha=0.5, color="red", label="Private")
df[df["Private"] == "No"]["Outstate"].plot.hist(alpha=0.5, color="blue", label="Public")
plt.legend()
plt.show()
'''
'''
g = sns.FacetGrid(data=df, hue="Private", size=6, aspect=2)
g = g.map(plt.hist, "Outstate", alpha=0.7)
plt.show()
'''
'''
g = sns.FacetGrid(data=df, hue="Private", size=6, aspect=2)
g = g.map(plt.hist, "Grad.Rate", alpha=0.7)
plt.show()
'''
X = df.drop(["Private"], axis=1)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

def convertPrivate(isPrivate: str):
   if isPrivate == "Yes":
      return 1
   else:
      return 0

def convertKmeans(temp: int):
   if temp == 0:
      return 1
   else:
      return 0
df["isPrivate"] = df["Private"].apply(convertPrivate)
df["kmeans"] = kmeans.labels_
df["kmeans2"] = df["kmeans"].apply(convertKmeans)

print(df)
print(len(df.columns))

nthGeneration = 0
for i in range(len(df)):
   if df.iloc[i, 18] == df.iloc[i, 20]:
      nthGeneration += 1
print(nthGeneration / len(df))
