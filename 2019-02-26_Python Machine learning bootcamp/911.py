import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("911.csv", parse_dates=["timeStamp"])
#print(df["zip"].value_counts().head(5))
#print(df["twp"].value_counts().head(5))
#print(df["title"].nunique())
df["reason"] = df["title"].apply(lambda text: text.split(sep=":")[0])
#sns.countplot(x="reason", data=df)
#plt.show()
df["dayofweek"] = df["timeStamp"].dt.dayofweek.map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
df["month"] = df["timeStamp"].dt.month

#sns.countplot(x="dayofweek", data=df, hue="reason")
#sns.countplot(x="month", data=df, hue="reason")
#plt.show()

byMonth = df.groupby("month")
#print(byMonth.count()["timeStamp"].plot())
#plt.show()
countByMonth = byMonth.count()
countByMonth = countByMonth.reset_index()
#sns.lmplot(x="month", y="twp", data=countByMonth)
#plt.show()

df["date"] = df["timeStamp"].dt.date
df["hour"] = df["timeStamp"].dt.hour
#df.groupby("date").count()["twp"].plot()
#plt.show()
df[df["reason"] == "Traffic"].groupby("date").count()["reason"].plot()
#plt.ylabel("Traffic counts")
#plt.show()

print(df)
hourByDayofweek = df.groupby(["dayofweek", "hour"]).count()["reason"].unstack(level=-1)
monthByDayofWeek = df.groupby(["dayofweek", "month"]).count()["reason"].unstack(level=-1)
#sns.heatmap(data=hourByDayofweek)
#sns.clustermap(data=hourByDayofweek)
sns.heatmap(data=monthByDayofWeek)
sns.clustermap(data=monthByDayofWeek)
plt.show()