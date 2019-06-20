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
from scipy.spatial.distance import correlation

sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion

print('loading dict files...')
folder = 'C:/Users/phan/OneDrive - adesso Group/DataSet/movielens-20m-dataset/'
with open(file=os.path.join(folder, 'user2movie.json'), mode='rb') as f:
   user2movie: Dict[int, List[int]] = pickle.load(f)
with open(file=os.path.join(folder, 'movie2user.json'), mode='rb') as f:
   movie2user: Dict[int, List[int]] = pickle.load(f)
with open(file=os.path.join(folder, 'userMovie2rating.json'), mode='rb') as f:
   userMovie2rating: Dict[Tuple[int, int], float] = pickle.load(f)
with open(file=os.path.join(folder, 'userMovie2ratingTest.json'), mode='rb') as f:
   userMovie2ratingTest: Dict[Tuple[int, int], float] = pickle.load(f)
with open(file=os.path.join(folder, 'movie2title.json'), mode='rb') as f:
   movie2title: Dict[Tuple[int, int], float] = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1  # number of users
m1 = np.max(list(movie2user.keys()))
m2 = np.max([movieId for (userId, movieId), rating in userMovie2ratingTest.items()])
M = max(m1, m2) + 1  # number of movies
print(f'{N} users ---- {M} movies')

# prepare train and test rating matrix
trainData = np.zeros(shape=(N, M))
for (userId, movieId), rating in userMovie2rating.items():
   trainData[userId, movieId] = rating
testData = np.zeros(shape=(N, M))
for (userId, movieId), rating in userMovie2ratingTest.items():
   testData[userId, movieId] = rating

# calculate pearson coeff
from sklearn.metrics.pairwise import pairwise_distances

userSimilarity = np.corrcoef(x=trainData, rowvar=True)


# userSimilarity2 = pairwise_distances(X=trainData, metric='correlation')

def predictUserBased(ratings: np.ndarray, similarity: np.ndarray):
   meanUserRatings = ratings.mean(axis=1)
   ratingDeviations = (ratings - meanUserRatings.reshape((N, 1)))
   return meanUserRatings.reshape((N, 1)) + similarity.dot(ratingDeviations) / np.abs(similarity).sum(
      axis=0).reshape((N, 1))


from sklearn.metrics import mean_squared_error


def rmse(prediction: np.ndarray, testData: np.ndarray):
   prediction = prediction[testData.nonzero()].flatten()
   testData = testData[testData.nonzero()].flatten()
   return np.sqrt(mean_squared_error(prediction, testData))


userBasedPrediction = predictUserBased(ratings=trainData, similarity=userSimilarity)
print(f'user based RMSE: {rmse(userBasedPrediction, testData)}')
