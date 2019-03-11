# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams['figure.figsize'] = (10, 4)
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

df = pd.read_csv("./data/P4-Section6-Homework-Dataset.csv", encoding="latin1", parse_dates=["Release Date"])

studio = ["Buena Vista Studios", "Sony", "Universal", "WB", "Paramount Pictures", "Fox"]
genre = ["action", "comedy", "adventure", "animation", "drama"]

df = df[df["Studio"].isin(studio) & df["Genre"].isin(genre)]

print(df)

f: plt.Figure = plt.figure(figsize=(10, 5))
sns.boxplot(x="Genre", y="Gross % US", data=df)
sns.stripplot(x="Genre", y="Gross % US", data=df, hue="Studio")
plt.show()
