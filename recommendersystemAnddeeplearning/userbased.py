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

KNeighbors = 25
LIMIT_LEAST_COMMON_MOVIES = 5
neighbors = []
averageRatings: List[float] = []
deviations: List[Dict[int, float]] = []

for userI in user2movie.keys():
   # find all movies rated by user i
   moviesOfUserI = user2movie[userI]
   moviesOfUserISet = set(moviesOfUserI)

   # calculate avg, deviations, sigma of user i
   movie2ratingOfI = {movieId: userMovie2rating[(userI, movieId)] for movieId in moviesOfUserI}
   avgRatingOfI = float(np.mean(list(movie2ratingOfI.values())))
   movie2devOfI = {movieId: (rating - avgRatingOfI) for movieId, rating in movie2ratingOfI.items()}
   devsOfIValues = np.array(list(movie2devOfI.values()))
   sigmaOfI = np.sqrt(devsOfIValues.dot(devsOfIValues))

   # print(movie2devOfI)

   # save for later use
   averageRatings.append(avgRatingOfI)
   deviations.append(movie2devOfI)

   neighborsOfUserI = SortedList()
   for userJ in user2movie.keys():
      if userJ != userI:
         moviesOfUserJ = user2movie[userJ]
         moviesOfUserJSet = set(moviesOfUserJ)
         commonMovies = (moviesOfUserISet & moviesOfUserJSet)

         if len(commonMovies) > LIMIT_LEAST_COMMON_MOVIES:
            # print(commonMovies)
            # print(len(commonMovies))
            # calculate avg, deviations and sigma of user j
            movie2ratingOfJ = {movieId: userMovie2rating[(userJ, movieId)] for movieId in moviesOfUserJ}
            avgRatingOfJ = float(np.mean(list(movie2ratingOfJ.values())))
            movie2devOfJ = {movieId: (rating - avgRatingOfJ) for movieId, rating in movie2ratingOfJ.items()}
            devsOfJValues = np.array(list(movie2devOfJ.values()))
            sigmaOfJ = np.sqrt(devsOfJValues.dot(devsOfJValues))

            # print(movie2devOfJ)

            # calculate the pearson correlation coefficient
            numerator = np.sum(([movie2devOfI[movieId] * movie2devOfJ[movieId] for movieId in commonMovies]))
            weightIJ = numerator / (sigmaOfI * sigmaOfJ)

            # add neighbors to sorted list
            neighborsOfUserI.add((-weightIJ, userJ))
            if len(neighborsOfUserI) > KNeighbors:
               del neighborsOfUserI[-1]

   # store the neighbors
   neighbors.append(neighborsOfUserI)
   print(f'processing user {userI}')
