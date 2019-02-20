import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

'''
menu = {"Filet Mignon": 29.99,
        "Big mag": 3.99,
        "Pizza": 0.99,
        "Salmon": 29.99}

print(menu.keys())
print(menu.get("Big magasd"))
'''
'''
iceScream = ["Chocolate", "Vanile", "Strawberry", "Rum Rainsin"]
newSerie = pd.Series(iceScream)
lottery = [4, 8, 15, 16, 23, 42]
lotSerie = pd.Series(lottery)
'''
'''
prices = [2.99, 4.45, 1.36]
s = pd.Series(prices)
print(s.mean())
'''
'''
fruits = ["Apple", "Orange", "Plum", "Grape", "Blueberry"]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
s = pd.Series(data=fruits, index=weekdays)
'''
'''
#pokemon = pd.read_csv("./data/pokemon.csv", usecols=["Pokemon"], squeeze=True)
pokemon = pd.read_csv("./data/pokemon.csv", index_col="Pokemon", squeeze=True)
google = pd.read_csv("./data/google_stock_price.csv", squeeze=True)

def classify(number):
   if number < 300:
      return "OK"
   elif number >= 300 and number < 650:
      return "Satisfactory"
   else:
      return "Incredible"
#print(google.apply(classify).value_counts())
print(google.apply(lambda price : price*10))
'''

nba = pd.read_csv("./data/nba.csv")
nba["Weight in Kilogram"] = nba["Weight"] * 0.453592

#nba = nba.dropna(subset=["Salary"])
#nba["College"] = nba["College"].fillna("No College")
#nba = nba.dropna()
#nba["Age"] = nba["Age"].astype(dtype="int64")
#nba["Team"] = nba["Team"].astype("category")
'''
nba = nba.sort_values(by=["Team", "Name"])
nba = nba.sort_index()
'''
nba = nba.dropna()
nba.insert(loc=9, column="Rank by Salary", value=0)
nba["Rank by Salary"] = nba["Salary"].rank(ascending=False).astype("int")
print(nba.sort_values(by="Rank by Salary"))