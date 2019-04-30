# region Import
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

frame = pd.read_csv('./rating_final.csv')
cuisine = pd.read_csv('./chefmozcuisine.csv')

ratingCount = pd.DataFrame(frame.groupby('placeID')['rating'].count())
ratingCount = ratingCount.sort_values(by='rating', ascending=False)

mostRatedPlace = ratingCount.loc[:132834, :]
summary = pd.merge(left=mostRatedPlace,
                   right=cuisine,
                   on='placeID')

print(summary)
print(mostRatedPlace)
