# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = 8, 4
sns.set()
plt.style.use("fivethirtyeight")
#pd.set_option('display.expand_frame_repr', False)
# endregion

df = sns.load_dataset("tips")
print(df.head(1))

#sns.distplot(a=df["total_bill"])
#sns.jointplot(data=df, x="total_bill", y="tip")
#sns.pairplot(data=df, hue="day")
#sns.jointplot(data=df, x="total_bill", y="tip", kind="reg")
#sns.scatterplot(data=df, x="total_bill", y="tip", hue="day")
#g = sns.FacetGrid(data=df, col="day", hue="sex")
#g = g.map(plt.scatter, "total_bill", "tip")
#plt.show()

#sns.barplot(data=df, x="sex", y="total_bill")
#sns.barplot(data=df, x="day", y="total_bill", estimator=np.mean)
#sns.countplot(data=df, x="sex")
#sns.boxplot(data=df, x="day", y="total_bill")
#sns.violinplot(data=df, x="day", y="total_bill", split=True, hue="sex")
#sns.stripplot(data=df, x="day", y="total_bill")
#sns.violinplot(data=df, x="day", y="total_bill")
#sns.swarmplot(data=df, x="day", y="total_bill", color="blue")
#sns.catplot(data=df, x="day", y="total_bill", kind="violin", hue="sex")
#sns.lmplot(data=df, x="total_bill", y="tip", hue="sex", markers=["o", "v"])
#sns.lmplot(data=df, x="total_bill", y="tip", col="sex", hue="sex", row="day", aspect=2)
#sns.distplot(a=df["total_bill"], norm_hist=False, kde=False)
#g = sns.FacetGrid(data=df, col="time", row="smoker")
#g.map(sns.distplot, "total_bill", norm_hist=False, kde=False)

#sns.set_context(context="poster")
#sns.countplot(data=df, x="sex")
sns.lmplot(data=df, x="total_bill", y="tip", hue="sex", fit_reg=False)
plt.show()


