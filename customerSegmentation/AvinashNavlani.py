# region Import
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import datetime as dt

sns.set()
pd.set_option('display.expand_frame_repr', False)
# endregion

filePickle = 'C:/Users/phan/OneDrive - adesso Group/DataSet/Online Retail.pickle'
with open(file=filePickle, mode='rb') as f:
   df = pickle.load(f)

df = df[df['Country'] == 'United Kingdom']
df = df[df['Quantity'] > 0]
df = df[['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
PRESENT = dt.datetime(2011, 12, 10)

rfm: pd.DataFrame = df.groupby('CustomerID').agg({
   'InvoiceDate': lambda date: (PRESENT - date.max()).days,
   'InvoiceNo': lambda num: len(num),
   'TotalPrice': lambda price: price.sum()
})

rfm.columns = ['recency', 'frequency', 'monetary']
qua = pd.qcut(x=range(100), q=4, labels=['1', '2', '3', '4'])

rfm['rQuartile'] = pd.qcut(x=rfm['recency'], q=4, labels=['1', '2', '3', '4'])
rfm['fQuartile'] = pd.qcut(x=rfm['frequency'], q=4, labels=['4', '3', '2', '1'])
rfm['mQuartile'] = pd.qcut(x=rfm['monetary'], q=4, labels=['4', '3', '2', '1'])