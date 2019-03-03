# region Import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from pandas_datareader import DataReader, data
from datetime import datetime
# endregion

techList = ["AAPL", "GOOG", "MSFT", "AMZN"]
endDate = datetime.now()
startDate = datetime(endDate.year - 1, endDate.month, endDate.day)

### Fetch data from finance source
'''
for stock in techList:
   globals()[stock] = DataReader(stock, "yahoo", startDate, endDate)
df = [AAPL, GOOG, MSFT, AMZN]
for i in range(len(df)):
   df[i].to_csv("./data/{}.csv".format(techList[i]))
'''

### Read data
df = []
for i in range(len(techList)):
   df.append(pd.read_csv("./data/{}.csv".format(techList[i])))

#AAPL["Adj Close"].plot(legend=True, figsize=(10, 4))
#plt.show()


maDay = [10, 20, 50]
for ma in maDay:
   columnName = "MA for {} days".format(ma)
   df[0][columnName] = df[0]["Adj Close"].rolling(window=ma).mean()
#AAPL[["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(figsize=(10, 6))
#plt.show()

#df[0]["Daily Return"] = df[0]["Adj Close"].pct_change()
#clgt = df[0].pct_change(1)


#df[0]["Daily Return"].plot(figsize=(10, 6), legend=True)
'''
plt.figure(figsize=(10, 6))
sns.distplot(df[0]["Daily Return"].dropna())
plt.show()
'''

#dfClosing = DataReader(techList, "yahoo", startDate, endDate)["Adj Close"]
#dfClosing.to_csv("dfClosing.csv")
dfClosing = pd.read_csv("./data/dfClosing.csv")
dfClosing = dfClosing.set_index(keys=["Date"])
techReturn = dfClosing.pct_change()

'''
#sns.jointplot(x="GOOG", y="GOOG", data=techReturn)
sns.pairplot(data=techReturn.dropna(), aspect=2)
plt.show()
'''

'''
fig = sns.PairGrid(data=techReturn.dropna())
fig.map_upper(plt.scatter)
fig.map_lower(sns.kdeplot)
fig.map_diag(plt.hist)
plt.show()
'''
'''
fig = sns.PairGrid(data=dfClosing.dropna())
fig.map_upper(plt.scatter)
fig.map_lower(sns.kdeplot)
fig.map_diag(plt.hist)
plt.show()
'''