# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

folder      = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/"
ratingFile  = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/rating.csv"
df = pd.read_csv(ratingFile)

# make user ids go start from 0
df['userId'] = df['userId'] - 1

# create mapping new movie id
uniqueMovieIds = set(df['movieId'].values)
movieIdDict = {}
newMovieIndex = 0
for movieId in uniqueMovieIds:
   movieIdDict[movieId] = newMovieIndex
   newMovieIndex += 1
print(movieIdDict)
# add to the data frame
df['movieIdNew'] = df['movieId'].apply(lambda idKey: movieIdDict.get(idKey))
df = df.drop(columns='timestamp')

# save to new file
print(df.head())
df.to_csv('{}editedRating.csv'.format(folder), index=None, columns=['userId', 'movieId', 'rating', 'movieIdNew'])

