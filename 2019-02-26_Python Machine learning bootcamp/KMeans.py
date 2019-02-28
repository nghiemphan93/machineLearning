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

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
'''
plt.scatter(data[0][:,0], data[0][:, 1], c=data[1])
plt.show()
'''

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 6))
ax1.set_title("K Means")
ax1.scatter(data[0][:,0], data[0][:, 1], c=kmeans.labels_)
ax2.set_title("Original")
ax2.scatter(data[0][:,0], data[0][:, 1], c=data[1])
plt.show()

correctCounter = 0
for i in range(len(data[1])):
   if data[1][i] == kmeans.labels_[i]:
      correctCounter = correctCounter + 1
#print(correctCounter/ len(data[1]))
print(correctCounter)
print(len(data[1]))
#print(metrics.accuracy_score(data[1], kmeans.labels_))