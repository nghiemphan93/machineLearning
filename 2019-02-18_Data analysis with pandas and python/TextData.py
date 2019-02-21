import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/chicago.csv")
df = df.dropna()
df["Position Title"] = df["Position Title"].str.title()
df["Name"] = df["Name"].str.title()
df["Department"] = df["Department"].str.title()
df["Employee Annual Salary"] = df["Employee Annual Salary"].str.replace("$", "").astype("float")

def extractFirstName(name):
    familyName, firstName  = name.split(sep=",")
    return firstName
def extractFamilyName(name):
    familyName, firstName = name.split(sep=",")
    return familyName
'''
df.insert(value=df["Name"].apply(extractFirstName), loc=1, column="First Name")
df.insert(value=df["Name"].apply(extractFamilyName), loc=2, column="Family Name")
'''

df[["First Name", "Last Name"]] = df["Name"].str.split(",", expand=True)
df.insert(loc=1, column="First Name", value=df.pop("First Name"))
df.insert(loc=2, column="Last Name", value=df.pop("Last Name"))

df[["First Title Word", "Remaining Words"]] = df["Position Title"].str.split(" ", expand=True, n=1)

print(df)
