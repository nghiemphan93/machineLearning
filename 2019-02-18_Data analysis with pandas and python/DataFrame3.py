import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/jamesbond.csv", index_col="Film")
df = df.sort_index()
#print(df.sample(n=10))
#print(df.loc[["Moonraker", "Octopussy"]])
#df.loc[df["Actor"] == "Sean Connery", "Actor"] = "clgt"
#df = df.drop(["Year", "Director"], axis="columns")
#print(df.nlargest(n=3, columns="Budget"))
#print(df.where(df["Box Office"] >= 800))
#print(df.query('Actor == "Sean Connery"'))
#print(df.query("Actor in ['Timothy Dalton', 'George Lazenby']"))
#df.columns = [colName.replace(" ", "_")  for colName in df.columns]

def convertToStringAndAddMillions(number):
    return str(number) + " Millions"
#df["Box Office"] = df["Box Office"].apply(convertToStringAndAddMillions)
'''
columns = ["Box Office", "Budget", "Bond Actor Salary"]
for colName in columns:
    df[colName] = df[colName].apply(convertToStringAndAddMillions)
'''

def goodMovie(row):
    actor = row[1]
    budget = row[4]

    if actor == "Pierce Brosnan":
        return "the best"
    elif actor == "Roger Moore" and budget > 40:
        return "enjoyable"
    else:
        return "I have no clue"
df["feedback"] = df.apply(goodMovie, axis="columns")
print(df)
