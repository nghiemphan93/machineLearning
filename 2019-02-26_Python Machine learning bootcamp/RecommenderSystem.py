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

column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("./data/u.data", sep="\t", names=column_names)
movie_titles = pd.read_csv("./data/Movie_Id_Titles")

df: pd.DataFrame = pd.merge(df, movie_titles, on="item_id")


#print(df.groupby("title")["rating"].mean().sort_values(ascending=False, axis=0))
#print(df.groupby("title")["rating"].count().sort_values(ascending=False))
ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
ratings["num of ratings"] = df.groupby("title")["rating"].count()

#print(ratings.sort_values(by=["num of ratings", "rating"], axis=0, ascending=False))
'''
sns.jointplot(x="rating", y="num of ratings", data=ratings)
plt.show()
'''

movieMat = df.pivot_table(index="user_id", columns="title", values="rating")

x = 10
print("clgt: {}".format(x))