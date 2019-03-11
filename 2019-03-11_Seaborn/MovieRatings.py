# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams['figure.figsize'] = (10, 4)
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = pd.read_csv("./data/P4-Movie-Ratings.csv")
df.columns = ["film", "genre", "criticRating", "audienceRating", "budgetMillions", "year"]
df["year"] = df["year"].astype("category")

print(df)

'''
sns.distplot(a=df["audienceRating"])
sns.jointplot(data=df, x="criticRating", y="audienceRating", kind="hex")
plt.show()
'''

#sns.lmplot(data=df, x="criticRating", y="audienceRating", hue="genre", legend=True, fit_reg=False, aspect=2)
#sns.lmplot(data=df, x="criticRating", y="audienceRating", hue="genre", legend=True, fit_reg=False, aspect=2)
'''
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df["criticRating"], data2=df["audienceRating"])
plt.show()
'''

'''
g = sns.FacetGrid(data=df, row="genre", col="year", hue="genre")
g = g.map(plt.scatter, "criticRating", "audienceRating")
plt.show()
'''
'''
g = sns.FacetGrid(data=df, row="genre", col="year", hue="genre")
g = g.map(plt.scatter, "criticRating", "audienceRating")
for ax in g.axes.flat:
   ax.plot((0, 100), (0, 100), c="gray", ls="--")
g.add_legend()
plt.show()
'''
