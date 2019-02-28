import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from pandas_datareader import data, wb
import datetime


'''
for bank in tickers:
   df: pd.DataFrame = data.DataReader(bank, "iex", start=start, end=end)
   df.to_csv()
'''
'''
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2019, 1, 1)

BAC = data.DataReader("BAC", 'iex', start, end)
C = data.DataReader("C", 'iex', start, end)
GS = data.DataReader("GS", 'iex', start, end)
JPM = data.DataReader("JPM", 'iex', start, end)
MS = data.DataReader("MS", 'iex', start, end)
WFC = data.DataReader("WFC", 'iex', start, end)
#df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'iex', start, end)
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
print(bank_stocks)
print(bank_stocks.info())
print(bank_stocks.xs("close", level="Stock Info", axis=1))
'''

start = "2015-01-01"
end = "2019-01-01"
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
BAC = pd.read_csv("BAC", index_col="date")
C = pd.read_csv("C", index_col="date")
GS = pd.read_csv("GS", index_col="date")
JPM = pd.read_csv("JPM", index_col="date")
MS = pd.read_csv("MS", index_col="date")
WFC = pd.read_csv("WFC", index_col="date")

df: pd.DataFrame = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
df.columns.names = ['Bank Ticker','Stock Info']
#print(df.xs(key="close", level="Stock Info", axis=1).max())

returns: pd.DataFrame = pd.DataFrame()
for ticker in tickers:
   returns[ticker + " returns"] = df.xs(key="close", level="Stock Info", axis=1)[ticker].pct_change()
print(returns)
returns = returns.dropna()
#sns.pairplot(data=returns)
#plt.show()

#for ticker in tickers:
#   print(returns[returns[ticker + " returns"] == returns[ticker + " returns"].min()])
'''
sns.distplot(returns.loc["2018-01-01":"2018-31-31"]["C returns"])
plt.title("C Returns in 2018")
plt.show()
'''
'''
plt.figure(figsize=(12, 8))
df.loc["2018-01-01":"2018-12-31"].xs(key="close", level="Stock Info", axis=1)["BAC"].plot(label="BAC Close")
df.loc["2018-01-01":"2018-12-31"].xs(key="close", level="Stock Info", axis=1)["BAC"].rolling(window=30).mean().plot(label="30 Day Average")
plt.legend()
plt.show()
'''

sns.heatmap(data=df.xs(key="close", level="Stock Info", axis=1).corr())
sns.clustermap(data=df.xs(key="close", level="Stock Info", axis=1).corr())
plt.show()


















