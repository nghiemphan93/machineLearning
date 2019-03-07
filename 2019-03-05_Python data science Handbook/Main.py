# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
# endregion

'''
#df = pd.read_csv("./data/president_heights.csv")
pop = pd.read_csv("./data/state-population.csv")
area = pd.read_csv("./data/state-areas.csv")
abbrevs = pd.read_csv("./data/state-abbrevs.csv")

print(pop)
print(area)
print(abbrevs)

df = pd.merge(left=pop, right=abbrevs, left_on="state/region", right_on="abbreviation")
df = pd.merge(left=df, right=area, on="state")
print(df)
'''
'''
planets = sns.load_dataset("planets")
print(planets)
sns.distplot(planets["mass"].dropna())
plt.show()
'''
'''
x = np.linspace(-5, 5, 100, dtype="float")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.Axes(ax[0]).hist()
plt.show()
'''
'''
fig: plt.Figure = plt.figure()
ax: plt.Axes = plt.axes()
x = np.linspace(0, 10, 10)
#ax.plot(x, np.sin(x), marker="o", label="sin(x)")
ax.scatter(x, np.sin(x), "x", label="sin(x)")
#ax.plot(x, np.cos(x), linestyle="--", label="cos(x)")
ax.set(xlabel="x", title="clgt")
plt.legend()
plt.show()
'''
'''
#fig: plt.Figure = plt.figure(nrows=2, ncols=2, figsize=(10, 4))
#ax: plt.Axes = plt.axes()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4), sharex="col", sharey="row")
x = np.random.normal(2, 3, 1000)
y = np.random.normal(0, 1, 1000)
#ax.hist(x, bins=30, alpha=0.5, label="big")
#ax.hist(y, bins=30, alpha=0.5, label="small")
ax[0, 0].hist(x, bins=30, alpha=0.5, color="red", label="big")
ax[0][0].legend()
ax[0][1].hist(y, bins=30, alpha=0.5, color="blue", label="small")
ax[0][1].legend()
#plt.Figure(fig).legend()
plt.show()
'''

df = sns.load_dataset("tips")
#print(df)
'''
#sns.pairplot(data=df, hue="sex")
plt.figure(figsize=(12, 6))
grid = sns.FacetGrid(data=df, row="sex", col="time", aspect=2, margin_titles=True)
grid.map(sns.distplot, "tip")
plt.show()
'''
'''
#g = sns.factorplot(x="day", y="total_bill", hue="sex", data=df, kind="box")
g = sns.jointplot(x="total_bill", y="tip", data=df, kind="reg")
g.set_axis_labels(xlabel="Total Bill", ylabel="Tips")
plt.show()
'''
'''
planets = sns.load_dataset("planets")
print(planets)
sns.catplot(x="year", data=planets, kind="count", aspect=2, color="steelblue", hue="method")
plt.show()
'''
'''
df = sns.load_dataset("iris")
model = PCA(n_components=2)
df2D = model.fit_transform(df.drop("species", axis=1))
df["PCA1"] = df2D[:, 0]
df["PCA2"] = df2D[:, 1]
print(df)
#sns.scatterplot(x="PCA1", y="PCA2", data=df, hue="species")
sns.lmplot(x="PCA1", y="PCA2", data=df, hue="species", fit_reg=False)
plt.show()
'''
'''
digits = load_digits()
df = pd.DataFrame(data=digits.data)
df["target"] = digits.target

pca = PCA(n_components=2)
digits2D = pca.fit_transform(X=digits.data)
df["PCA1"] = digits2D[:, 0]
df["PCA2"] = digits2D[:, 1]
sns.lmplot(x="PCA1", y="PCA2", data=df, hue="target", fit_reg=False)
plt.show()
'''

X, y = make_moons(200, noise=0.05, random_state=0)
model = SpectralClustering(n_clusters=2,
                           affinity="nearest_neighbors",
                           assign_labels="kmeans")
yPredicted = model.fit_predict(X)
plt.scatter(x=X[:, 0], y=X[:, 1], c=yPredicted)
plt.show()