# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 8, 4
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = sns.load_dataset("titanic")
print(df.head(5))

#sns.distplot(a=df["fare"], norm_hist=False, kde=False)
#sns.boxplot(data=df, x="class", y="age")
#sns.violinplot(data=df, x="class", y="age", palette="rainbow")
#sns.swarmplot(data=df, x="class", y="age", palette="rainbow")
#sns.heatmap(data=df.corr())
#sns.clustermap(data=df.corr())
g = sns.FacetGrid(data=df, col="sex")
g.map(sns.distplot, "age")
plt.show()