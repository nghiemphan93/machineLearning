import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan

Xmoon, ymoon = make_moons(1000, noise=.05, random_state=0)
df = pd.DataFrame(data=Xmoon, columns=["x", "y"])

# model = KMeans(n_clusters=2)
# model = DBSCAN(eps=0.3)
# model = GaussianMixture(n_components=2, covariance_type="diag")

for i in range(2, 200, 1):
   model = hdbscan.HDBSCAN(min_cluster_size=i)
   model.fit(Xmoon)
   print("i: ", i, " class: ", len(np.unique(model.labels_)))


# sns.scatterplot(data=df, x="x", y="y", hue=model.labels_)
# plt.show()

# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
# plt.show()
# print(df)
