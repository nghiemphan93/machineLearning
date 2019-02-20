import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas import Series, DataFrame

df = pd.read_csv("./data/employees.csv", parse_dates=["Start Date", "Last Login Time"])
#df["Start Date"] = pd.to_datetime(df["Start Date"])
#df["Last Login Time"] = pd.to_datetime(df["Last Login Time"])
df["Senior Management"] = df["Senior Management"].astype(dtype="bool")
df["Gender"] = df["Gender"].astype(dtype="category")
df["Team"] = df["Team"].astype(dtype="category")

#print(df[df["Gender"] == "Male" or df["Gender"] == "Female"])
isFinance = df["Team"] == "Finance"
isNotFinance = df["Team"] != "Finance"
vor1985 = df["Start Date"] <= "01.01.1985"

isRobert          = df["First Name"] == "Robert"
isClientServices  = df["Team"] == "Client Services"
isAfter01062016   = df["Start Date"] > "01.06.2016"
isInTeam          = df["Team"].isin(["Legal", "Sales", "Product"])
isBetween         = df["Salary"].between(left=60000, right=70000, inclusive=True)

#print(df[isBetween].sort_values(by="Salary", ascending=False))
df = df.sort_values("First Name")
#print(df[~df["First Name"].duplicated(keep=False)])
#print(df.drop_duplicates(["First Name"], keep=False).drop_duplicates("Team", keep=False))
print(df["First Name"].unique())
