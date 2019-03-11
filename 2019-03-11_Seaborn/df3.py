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

df = pd.read_csv("./data/df3")
print(df.head(5))

plt.figure(figsize=(6, 4))
#sns.scatterplot(data=df, x="a", y="b")
#sns.distplot(a=df["a"], kde=False, norm_hist=False)
plt.show()