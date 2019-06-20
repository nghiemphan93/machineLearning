# region Import
import os, pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
from sortedcontainers import SortedList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import tmdbsimple as tmdb
import urllib


sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion

columns = ['userId', 'itemId', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', names=columns, sep='\t')

movies = pd.read_csv('./ml-100k/u.item', sep='|', encoding='latin', names=range(24))
movies = movies.iloc[:, :2]
movies['year'] = movies.iloc[:, 1].str[-5:-1]
movies.iloc[:, 1] = movies.iloc[:, 1].str[:-6]
movies = movies.rename(columns={0: 'movieId', 1: 'title'})

N_USERS = df['userId'].nunique()
N_ITEMS = df['itemId'].nunique()

# df.loc[:, 'userId'] = df['userId'] - 1
# df.loc[:, 'itemId'] = df['itemId'] - 1

ratings = np.zeros(shape=(N_USERS, N_ITEMS))

# def createRatingsMatrix(row: pd.Series):
#    ratings[row['userId'] - 1, row['itemId'] - 1] = row['rating']
#    print(row['userId'])


# df.apply(createRatingsMatrix, axis=1)

# Create ratings matrix
for row in df.itertuples():
   ratings[row[1] - 1, row[2] - 1] = row[3]

sparsity = float(len(ratings.nonzero()[0])) / (ratings.shape[0] * ratings.shape[1]) * 100


def trainTestSplit(ratings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
   """
   Take 10 ratings of a movie of each user to the test set
   The remainder stays in the train set
   :param ratings:
   :return:
   """

   test = np.zeros(shape=ratings.shape)
   train = ratings.copy()

   for userIndex in range(ratings.shape[0]):
      testUserIndex = np.random.choice(a=ratings[userIndex, :].nonzero()[0],
                                       size=10,
                                       replace=False)
      train[userIndex, testUserIndex] = 0
      test[userIndex, testUserIndex] = ratings[userIndex, testUserIndex]
   return train, test


train, test = trainTestSplit(ratings=ratings)

from sklearn.metrics import pairwise_distances

# userSimilarity = 1 - pairwise_distances(X=train, metric='cosine')
# itemSimilarity = 1 - pairwise_distances(X=train.T, metric='cosine')


userSimilarity = 1 - pairwise_distances(X=train, metric='correlation')
itemSimilarity = 1 - pairwise_distances(X=train.T, metric='correlation')
itemSimilarity[np.isnan(itemSimilarity)] = 0


def predictUserRatingSimple(ratings: np.ndarray, userSimilarity: np.ndarray):
   return userSimilarity.dot(ratings) / np.abs(userSimilarity).sum(axis=1).reshape((N_USERS, 1))


def predictItemRatingSimple(ratings: np.ndarray, itemSimilarity: np.ndarray):
   return itemSimilarity.dot(ratings) / np.abs(itemSimilarity).sum(axis=1).reshape((N_ITEMS, 1))


from sklearn.metrics import mean_squared_error


def getMSE(predicted: np.ndarray, actual: np.ndarray):
   predicted = predicted[actual.nonzero()].flatten()
   actual = actual[actual.nonzero()].flatten()
   return mean_squared_error(predicted, actual)


userPrediction = predictUserRatingSimple(train, userSimilarity)
itemPrediction = predictItemRatingSimple(train.T, itemSimilarity)


print(f'User-based CF MSE: {getMSE(userPrediction, test)}')
print(f'Item-based CF MSE: {getMSE(itemPrediction, test.T)}')


def predictTopKUserBased(ratings: np.ndarray, userSimilarity: np.ndarray, kUsers=40) -> np.ndarray:
   prediction = np.zeros(shape=ratings.shape)
   for userIndex in range(ratings.shape[0]):
      print(f'working on user {userIndex}')
      topKUsersIndices = np.argsort(userSimilarity[userIndex])[::-1][:kUsers]
      for itemIndex in range(ratings.shape[1]):
         nominator = userSimilarity[userIndex, topKUsersIndices].dot(ratings[topKUsersIndices, itemIndex])
         denominator = np.sum(np.abs(userSimilarity[userIndex, topKUsersIndices]))
         prediction[userIndex, itemIndex] = nominator / denominator
   return prediction


def predictTopKItemBased(ratings: np.ndarray, itemSimilarity: np.ndarray, kItems=40) -> np.ndarray:
   prediction = np.zeros(shape=ratings.shape)
   for i in range(ratings.shape[0]):
      print(f'working on item {i}')
      topKItemsIndices = np.argsort(itemSimilarity[i])[::-1][:kItems]
      for j in range(ratings.shape[1]):
         nominator = itemSimilarity[i, topKItemsIndices].dot(ratings[topKItemsIndices, j])
         denominator = np.sum(np.abs(itemSimilarity[i, topKItemsIndices]))
         prediction[i, j] = nominator / denominator
   return prediction


userPredictionTopK = predictTopKUserBased(train, userSimilarity)
itemPredictionTopK = predictTopKItemBased(train.T, itemSimilarity)

print(f'Top-k User-based CG MSE: {getMSE(userPredictionTopK, test)}')
print(f'Top-k Item-based CG MSE: {getMSE(itemPredictionTopK, test.T)}')


def predictNoBiasUserBased(ratings: np.ndarray, userSimilarity: np.ndarray):
   meanRating = ratings.mean(axis=1).reshape(N_USERS, 1)
   deviationRating = ratings - meanRating
   return meanRating + userSimilarity.dot(deviationRating) / np.abs(userSimilarity).sum(axis=1).reshape(N_USERS, 1)


def predictNoBiasItemBased(ratings: np.ndarray, itemSimilarity: np.ndarray):
   meanRating = ratings.mean(axis=1).reshape(N_ITEMS, 1)
   deviationRating = ratings - meanRating
   return meanRating + itemSimilarity.dot(deviationRating) / np.abs(itemSimilarity).sum(axis=1).reshape(N_ITEMS, 1)


userPredictionNoBias = predictNoBiasUserBased(train, userSimilarity)
itemPredictionNoBias = predictNoBiasItemBased(train.T, itemSimilarity)

print(f'NoBias User-based CG MSE: {getMSE(userPredictionNoBias, test)}')
print(f'NoBias Item-based CG MSE: {getMSE(itemPredictionNoBias, test.T)}')


def predictTopKNoBiasUserBased(ratings: np.ndarray, userSimilarity: np.ndarray, k=40):
   prediction = np.zeros(shape=ratings.shape)
   meanRating = ratings.mean(axis=1).reshape(N_USERS, 1)
   deviationRating = ratings - meanRating

   for userIndex in range(deviationRating.shape[0]):
      print(f'working on user {userIndex}')
      topKUserIndices = np.argsort(userSimilarity[userIndex])[::-1][:k]
      for itemIndex in range(deviationRating.shape[1]):
         nominator = userSimilarity[userIndex, topKUserIndices].dot(deviationRating[topKUserIndices, itemIndex])
         denominator = np.sum(np.abs(userSimilarity[userIndex, topKUserIndices]))
         prediction[userIndex, itemIndex] = nominator / denominator
   prediction += meanRating
   return prediction


def predictTopKNoBiasItemBased(ratings: np.ndarray, itemSimilarity: np.ndarray, k=40):
   prediction = np.zeros(shape=ratings.shape)
   meanRating = ratings.mean(axis=1).reshape(N_ITEMS, 1)
   deviationRating = ratings - meanRating

   for itemIndex in range(deviationRating.shape[0]):
      print(f'working on item {itemIndex}')
      topKItemIndices = np.argsort(itemSimilarity[itemIndex])[::-1][:k]
      for userIndex in range(deviationRating.shape[1]):
         nominator = itemSimilarity[itemIndex, topKItemIndices].dot(deviationRating[topKItemIndices, userIndex])
         denominator = np.sum(np.abs(itemSimilarity[itemIndex, topKItemIndices]))
         prediction[itemIndex, userIndex] = nominator / denominator
   prediction += meanRating
   return prediction


userPredictionTopKNoBias = predictTopKNoBiasUserBased(train, userSimilarity)
itemPredictionTopKNoBias = predictTopKNoBiasItemBased(train.T, itemSimilarity)

print(f'Top K NoBias User-based CG MSE: {getMSE(userPredictionTopKNoBias, test)}')
print(f'Top K NoBias Item-based CG MSE: {getMSE(itemPredictionTopKNoBias, test.T)}')

def showPoster(movieId: int):
   tmdb.API_KEY = '44323693051c3a17e0ded73a75aba1be'
   movieTile = movies.loc[movieId, 'title']
   search = tmdb.Search()
   response = search.movie(query=movieTile)
   moviePosterLink = f'https://image.tmdb.org/t/p/w500/{search.results[0]["poster_path"]}'
   moviePosterImg = urllib.request.urlopen(moviePosterLink)
   moviePosterImg = plt.imread(moviePosterImg, format='jpg')
   plt.imshow(X=moviePosterImg)
   plt.show()


def getTopKMovieIndices(itemSimilarity: np.ndarray, similarToMovieId: int, k=6):
   return np.argsort(itemSimilarity[similarToMovieId])[::-1][:k]


similarMovies = getTopKMovieIndices(itemSimilarity, 1)
for movieId in similarMovies:
   showPoster(movieId)