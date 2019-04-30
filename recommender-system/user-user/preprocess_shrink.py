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

folder      = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/"
ratingFileEdited  = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/editedRating.csv"
df = pd.read_csv(ratingFileEdited)

# number of users and movies
N = df['userId'].max() + 1
M = df['movieIdNew'].max() + 1

# times appeared
userIdsCount = Counter(df['userId'])
movieIdsCount = Counter(df['movieIdNew'])

# number of users and movies to keep
nUsers = 10000   # users
mMovies = 2000    # movies

# users and movies to be kept
userIds  = [userId for userId, frequency in userIdsCount.most_common(nUsers)]
movieIds = [movieId for movieId, frequency in movieIdsCount.most_common(mMovies)]

# make a copy
dfSmall = df[(df['userId'].isin(userIds)) & (df['movieIdNew'].isin(movieIds))].copy()

# make new userIds and movieIds
newUserIdDict = {}
userCounter = 0
for userId in userIds:
   newUserIdDict[userId] = userCounter
   userCounter += 1

newMovieIdDict = {}
movieCounter = 0
for movieId in movieIds:
   newMovieIdDict[movieId] = movieCounter
   movieCounter += 1

dfSmall['movieID'] = dfSmall['movieIdNew'].apply(lambda id: newMovieIdDict.get(id))
dfSmall['userID'] = dfSmall['userId'].apply(lambda id: newUserIdDict.get(id))
dfSmall: pd.DataFrame = dfSmall[['userID', 'movieID', 'rating']]
dfSmall = dfSmall.sort_values(by='userID').copy()
dfSmall = dfSmall.reset_index()

print(dfSmall.head())
dfSmall.to_csv('{}verySmallRating.csv'.format(folder), index=None, columns=['userID', 'movieID', 'rating'])


