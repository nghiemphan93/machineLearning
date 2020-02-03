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
   df: pd.DataFrame = pickle.load(f)

# df.groupby('Country').agg({'CustomerID': lambda customer: len(customer)}).sort_values('CustomerID', ascending=False)

df = df[df['Country'] == 'United Kingdom']
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

NOW = dt.datetime(2011, 12, 10)

rfm: pd.DataFrame = df.groupby('CustomerID').agg({
   'InvoiceDate': lambda dates: (NOW - dates.max()).days,
   'InvoiceNo': lambda invoices: len(invoices),
   'TotalPrice': lambda prices: sum(prices)
})

rfm = rfm.rename(columns={
   'InvoiceDate': 'recency',
   'InvoiceNo': 'frequency',
   'TotalPrice': 'monetary'
})

rfm['rQuartile'] = pd.qcut(x=rfm['recency'], q=4, labels=[1, 2, 3, 4])
rfm['fQuartile'] = pd.qcut(x=rfm['frequency'], q=4, labels=[4, 3, 2, 1])
rfm['mQuartile'] = pd.qcut(x=rfm['monetary'], q=4, labels=[4, 3, 2, 1])

rfm['rfmScore'] = rfm['rQuartile'].astype('str') + rfm['fQuartile'].astype('str') + rfm['mQuartile'].astype('str')
