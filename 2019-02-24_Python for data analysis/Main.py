import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import numpy as np

df = pd.read_csv("./police.csv")

# Remove the column that only contains missing values
#df = df.dropna(how="all", axis="columns")
df = df.drop(columns="county_name", axis="columns")

# Do men or women speed more often?
#print(df[df["violation"] == "Speeding"]["driver_gender"].value_counts(normalize=True) * 100)
#print(df.groupby(by="driver_gender")["violation"].value_counts())

# Does gender affect who gets searched during a stop?
#print(df[df["search_conducted"] == True]["driver_gender"].value_counts(normalize=True))
#print(df.groupby(by="driver_gender")["search_conducted"].mean())

# During a search, how often is the driver frisked?
#print(df[df["search_conducted"] == True])
#print(df["search_type"].str.contains("Protective Frisk").value_counts(normalize=True))

# Which year has the least number of stops?
#print(df["stop_date"].str.slice(0, 4).value_counts())
#df["stop_date"] = pd.to_datetime(df["stop_date"])
df["combined"] = df["stop_date"].str.cat(df["stop_time"], sep=" ")
df["stop_datetime"] = pd.to_datetime(df["combined"])
#print(df["stop_datetime"].dt.year.value_counts())

# How does drug activity change by time of day
#print(df["drugs_related_stop"].value_counts())
#print(df[df["drugs_related_stop"] == True].groupby(df["stop_datetime"].dt.hour)["drugs_related_stop"].sum())
#df[df["drugs_related_stop"] == True].groupby(df["stop_datetime"].dt.hour)["drugs_related_stop"].sum().plot(kind="bar")
#plt.show()

# Do most stops occur at night?
#df["stop_datetime"].dt.hour.value_counts().sort_index().plot(kind="bar")
#plt.show()

# Find the bad data in the stop_duration column and fix it
df.loc[(df["stop_duration"] == "1") | (df["stop_duration"] == "2"), "stop_duration"] = np.nan
#print(df["stop_duration"].value_counts())

