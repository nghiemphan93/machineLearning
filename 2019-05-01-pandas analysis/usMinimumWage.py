# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = pd.read_csv('./dataset/Minimum Wage Data.csv', encoding='latin')
df = df.set_index(keys='Year')
print(df.head())


gb = df.groupby(by='State')
actualMinWage = pd.DataFrame()
for stateName, group in df.groupby(by='State'):
   group = group.rename(columns={'Low.2018': stateName})
   if actualMinWage.empty:
      actualMinWage = pd.DataFrame(group[stateName])
   else:
      actualMinWage = actualMinWage.join(other=group[stateName])

actualMinWage = actualMinWage.replace(0, np.nan).dropna(axis=1)
print(actualMinWage.head())

# print(df.pivot_table(index='Year', columns='State', values='Low.2018').head())
