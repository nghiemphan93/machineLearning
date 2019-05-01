# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = pd.read_csv('./dataset/avocado.csv', parse_dates=['Date'])
df = df.set_index(keys=['Date'])
df = df.drop(columns=['Unnamed: 0'])
df = df.sort_index()
# df['type'] = df['type'].astype('category')
print(df.head())


# chicago: pd.DataFrame = df[df['region'] == 'Chicago'].copy()
# chicago['price25ma'] = chicago['AveragePrice'].rolling(window=25).mean()
# print(chicago['price25ma'].head(50))

graphDF = pd.DataFrame()
for region in df['region'].unique():
   regionDf: pd.DataFrame = df[(df['region'] == region) & (df['type'] == 'organic')].copy()
   regionDf['{}-price25mean'.format(region)] = regionDf['AveragePrice'].rolling(window=25).mean()
   regionDf = regionDf.dropna()

   if graphDF.empty:
      graphDF = pd.DataFrame(regionDf['{}-price25mean'.format(region)])
   else:
      graphDF = graphDF.join(other=regionDf['{}-price25mean'.format(region)])
   # print(region)
print(graphDF.head())

graphDF['Southeast-price25mean'].plot()
plt.show()