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
import nltk, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download_shell()
'''
messages = [line.rstrip() for line in open("./data/SMSSpamCollection")]
for messIndex, mess in enumerate(messages[:10]):
   print(messIndex, mess)
'''
df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["label", "message"])
#print(df.groupby("label").describe())
df["length"] = df["message"].apply(len)

'''
plt.figure(figsize=(12, 5))
sns.distplot(df[df["label"] == "spam"]["length"], bins=30, color="red", label="Spam")
sns.distplot(df[df["label"] == "ham"]["length"], bins=30, color="blue", label="Ham")
plt.legend()
plt.show()
'''
mess = "jfDewf jaiEwef. aweHj? jioaef !"
nopunc = [c for c in mess if c not in string.punctuation]
nopunc = "".join(nopunc)


def textProcess(mess):
   # remove punc
   # remove stop words
   # return list of clean text words
   nopunc = [char for char in mess if char not in string.punctuation]
   nopunc = "".join(nopunc)
   return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

#print(df["message"].apply(textProcess))
#print(df["message"].head().apply(textProcess))

bowTransformer = CountVectorizer(analyzer=textProcess).fit(df["message"])

