import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)

tips = sns.load_dataset("tips")
'''
sns.distplot(tips["total_bill"])
sns.pairplot(tips)
plt.show()
'''
'''
#sns.barplot(x="sex", y="total_bill", data=tips)
sns.boxplot(x="day", y="total_bill", data=tips, )
plt.show()
'''
df = sns.load_dataset("titanic")
print(df)
#sns.jointplot(x="fare", y="age", data=df)
#sns.distplot(a=df["fare"], kde=False)
#sns.boxplot(x="class", y="age", data=df)
#sns.swarmplot(x="class", y="age", data=df)
#sns.countplot(x="sex", data=df)
print(df.corr())
plt.show()
