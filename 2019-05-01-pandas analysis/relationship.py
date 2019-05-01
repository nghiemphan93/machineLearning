# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

minWage = pd.read_csv('./dataset/Minimum Wage Data.csv', encoding='latin')
minWage = minWage.set_index('Year')
minWage = pd.pivot_table(data=minWage, index='Year', columns='State', values='Low.2018')
minWage = minWage.replace(0, np.nan).dropna(axis=1)

unemCounty = pd.read_csv('./dataset/output.csv')
unemCounty = unemCounty.set_index('Year')

print(unemCounty.head())
print(minWage.head())