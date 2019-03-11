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

df = sns.load_dataset("flights")
print(df.head(1))

df = df.pivot_table(index="month", columns="year", values="passengers")
print(df.head(10))

#sns.heatmap(data=df, annot=False)
#sns.clustermap(data=df, cmap="coolwarm", standard_scale=1)
plt.show()