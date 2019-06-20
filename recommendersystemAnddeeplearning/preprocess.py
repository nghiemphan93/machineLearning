# region Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion


folder = 'C:/Users/phan/OneDrive - adesso Group/DataSet/movielens-20m-dataset/'
rating = os.path.join(folder, 'rating.csv')
movie = os.path.join(folder, 'movie.csv')
df = pd.read_csv(rating)
movie = pd.read_csv(movie)


# make user id go from 0
df['userId'] = df['userId'] - 1

# rearrange mapping for movie ids sequentially from 0
uniqueMovieIds = set(df['movieId'].values)
movie2index = {}
count = 0
for movieId in uniqueMovieIds:

   movie2index[movieId] = count
   count += 1

# add to the data frame
df['newMovieId'] = df['movieId'].apply(lambda movieId: movie2index[movieId])
# df['newMovieId'] = df.apply(lambda row: movie2index[row.movieId], axis=1)


newDf = pd.merge(left=df, right=movie, on='movieId')
newDf = newDf[['userId', 'movieId', 'rating', 'newMovieId', 'title']]
newDf.to_csv(os.path.join(folder, 'editedRating.csv'))