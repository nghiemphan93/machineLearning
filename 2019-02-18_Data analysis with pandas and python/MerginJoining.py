import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

week1 = pd.read_csv("./data/Restaurant - Week 1 Sales.csv")
week2 = pd.read_csv("./data/Restaurant - Week 2 Sales.csv")
customers = pd.read_csv("./data/Restaurant - Customers.csv")
foods = pd.read_csv("./data/Restaurant - Foods.csv")

sales = pd.concat(objs=[week1, week2],  keys=["one", "two"])
#print(week1.merge(right=week2, how="inner", on="Customer ID", suffixes=[" Week 1", " Week 2"]))
print(week1.merge(right=week2, how="inner", on=["Customer ID", "Food ID"], suffixes=["-A", "-B"]))