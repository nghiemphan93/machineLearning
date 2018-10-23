from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame


df: DataFrame = pd.read_csv("./data/police.csv")
df["stop_datetime"] = df.stop_date.str.cat(df.stop_time, " ")
df["stop_datetime"] = pd.to_datetime(df["stop_datetime"])


# Remove columns that only contains missing values
'''
print(df.info())
#df = df.drop("county_name", axis="columns")
df = df.dropna(axis="columns", how="all")
print(df.info())
'''


# Do men or women speed more often?
'''
print(df.loc[df["violation"] == "Speeding"]["driver_gender"].value_counts())
df.loc[df["violation"] == "Speeding"]["driver_gender"].value_counts().plot(kind="bar")
plt.show()
'''

'''
print(df[df["driver_gender"] == "M"]["violation"].value_counts(normalize=True))
print(df[df["driver_gender"] == "F"]["violation"].value_counts(normalize=True))
'''
'''
print(df.groupby("driver_gender")["violation"].value_counts(normalize=True).unstack())
print(type(df.groupby("driver_gender")["violation"].value_counts(normalize=True)))
'''

# Gender vs Search
'''
print(df.groupby("driver_gender").search_conducted.mean())
print(df.groupby(["violation", "driver_gender"]).search_conducted.mean())
'''

# Why so many NaN in search_type?
'''
print(df[df.search_conducted == False].search_type.value_counts(dropna=False))
print(df.search_type.value_counts(dropna=False))
'''

# During a seach, how often is the driver frisked?
'''
df["frisk"] = df.search_type.str.contains("Protective Frisk")
print(df.frisk.value_counts(normalize=True))
'''

# Which year had the least number of stops?
'''
print(df.stop_date.str.slice(0, 4).value_counts())
df.stop_date = pd.to_datetime(df.stop_date)
print(df.stop_date.dt.year.value_counts())
print(df.stop_date.dt.year.value_counts().sort_values().index[0])
'''

# How does drug activity change by time of day
'''
df["stop_datetime"] = df.stop_date.str.cat(df.stop_time, " ")
df["stop_datetime"] = pd.to_datetime(df["stop_datetime"])

#print(df.groupby(df.stop_datetime.dt.hour).drugs_related_stop.mean())
#dfNew = df.groupby(df.stop_datetime.dt.hour).drugs_related_stop.value_counts()
#df.groupby(df.stop_datetime.dt.hour).drugs_related_stop.mean().plot(kind="bar")
#plt.show()

df = df.groupby(df.stop_datetime.dt.hour).drugs_related_stop.value_counts().unstack()
df.iloc[:, 1].plot()
plt.show()
'''

# Do most stops occur at night
'''
#print(df.stop_datetime.dt.hour.value_counts().sort_index())
print(df[(df.stop_datetime.dt.hour < 4) | (df.stop_datetime.dt.hour > 22)].shape)
#df.stop_datetime.dt.hour.value_counts().sort_index().plot()
plt.show()
'''

# Find the bad data in the stop_duration column and fix it
print(df.stop_duration.value_counts())
df.loc[(df.stop_duration == "1") | (df.stop_duration == "2"), "stop_duration"] = np.nan
print(df.stop_duration.value_counts())