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
geodata = pd.read_csv('./geoplaces2.csv', encoding='latin-1')
places = geodata[['placeID', 'name']]
rating = pd.DataFrame(frame.groupby(by='placeID')['rating'].mean())
rating['ratingCount'] = pd.DataFrame(frame.groupby(by='placeID')['rating'].count())
rating = rating.sort_values(by='ratingCount', ascending=False)
placesCrossTab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
tortasRatings = placesCrossTab[135085]
similarToTortas = placesCrossTab.corrwith(tortasRatings)
corrTortas = pd.DataFrame(similarToTortas, columns=['pearsonR'])
corrTortas = corrTortas.dropna()

print(places.head())
print(cuisine.head())
print(frame.head())
print(rating.head())
print(cuisine[cuisine['placeID'] == 135085])
print(placesCrossTab)
print(corrTortas)