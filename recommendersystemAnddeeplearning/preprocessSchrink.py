# region Import
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion

folder = 'C:/Users/phan/OneDrive - adesso Group/DataSet/movielens-20m-dataset/'
editedRating = os.path.join(folder, 'editedRating.csv')
df = pd.read_csv(editedRating)

N = df['userId'].max() + 1
M = df['newMovieId'].max() + 1

userIdsCount = Counter(df['userId'])
movieIdsCount = Counter(df['newMovieId'])

# number of user and movies to keep
n = 10000
m = 2000

userIds = [userId for userId, count in userIdsCount.most_common(n)]
movieIds = [movieId for movieId, count in movieIdsCount.most_common(m)]

# make a copy of the users and movies to keep
dfSmall = df[(df['userId'].isin(userIds)) & (df['newMovieId'].isin(movieIds))].copy()
dfSmall = dfSmall.drop('Unnamed: 0', axis=1)
dfSmall = dfSmall.drop('movieId', axis=1)

# rearrange new userIds and movieIds
oldUserId2newUserId = {}
count = 0
for oldUserId in userIds:
   oldUserId2newUserId[oldUserId] = count
   count += 1

oldMovieId2newMovieId = {}
count = 0
for oldMovieId in movieIds:
   oldMovieId2newMovieId[oldMovieId] = count
   count += 1

# Aplly new ids to old ids in data frame
print('setting new ids...')
dfSmall.loc[:, 'userId'] = dfSmall['userId'].apply(lambda oldUserId: oldUserId2newUserId[oldUserId])
dfSmall.loc[:, 'newMovieId'] = dfSmall['newMovieId'].apply(lambda oldMovieId: oldMovieId2newMovieId[oldMovieId])

print('writing small dataframe...')
dfSmall = dfSmall.rename(columns={'newMovieId': 'movieId'})
dfSmall = pd.DataFrame(data=dfSmall, columns=['userId', 'movieId', 'rating', 'title'])
dfSmall.to_csv(os.path.join(folder, 'verySmallRating.csv'), index=False)