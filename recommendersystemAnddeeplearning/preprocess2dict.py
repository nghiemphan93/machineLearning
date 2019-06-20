# region Import
import os, pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion

folder = 'C:/Users/phan/OneDrive - adesso Group/DataSet/movielens-20m-dataset/'
smallRating = os.path.join(folder, 'verySmallRating.csv')
df = pd.read_csv(smallRating)

N = df['userId'].max() + 1  # number of users
M = df['movieId'].max() + 1  # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8 * len(df))
dfTrain = df.iloc[:cutoff]
dfTest = df.iloc[cutoff:]

# create 3 dicts for lookup
user2movie: Dict[int, List[int]] = defaultdict(list)
movie2user: Dict[int, List[int]] = defaultdict(list)
userMovie2rating: Dict[Tuple[int, int], float] = defaultdict(float)
movie2title: Dict[int, str] = defaultdict(str)

print('calling map user2movie and movie2user...')
count = 0


def mapUser2movieAndMovie2User(row: pd.Series):
   global count
   count += 1
   if count % 50000 == 0:
      print(f'processed: {count / cutoff * 100:.3f} %')

   userId = int(row['userId'])
   movieId = int(row['movieId'])
   rating = int(row['rating'])
   title = str(row['title'])

   user2movie[userId].append(movieId)
   movie2user[movieId].append(userId)
   userMovie2rating[(userId, movieId)] = rating
   movie2title[movieId] = title


dfTrain.apply(lambda row: mapUser2movieAndMovie2User(row), axis=1)

# test ratings dict
userMovie2ratingTest: Dict[Tuple[int, int], float] = defaultdict(float)
count = 0


def mapUserMovie2ratingTest(row: pd.Series):
   global count
   count += 1
   if count % 50000 == 0:
      print(f'processed: {count / cutoff * 100:.3f} %')
   userId = int(row['userId'])
   movieId = int(row['movieId'])
   rating = int(row['rating'])
   title = str(row['title'])
   userMovie2ratingTest[(userId, movieId)] = rating
   movie2title[movieId] = title


dfTest.apply(lambda row: mapUserMovie2ratingTest(row), axis=1)

# save dicts to pickle files
print('writing dicts to pickle files...')
with open(file=os.path.join(folder, 'user2movie.json'), mode='wb') as f:
   pickle.dump(user2movie, f)
with open(file=os.path.join(folder, 'movie2user.json'), mode='wb') as f:
   pickle.dump(movie2user, f)
with open(file=os.path.join(folder, 'userMovie2rating.json'), mode='wb') as f:
   pickle.dump(userMovie2rating, f)
with open(file=os.path.join(folder, 'userMovie2ratingTest.json'), mode='wb') as f:
   pickle.dump(userMovie2ratingTest, f)
with open(file=os.path.join(folder, 'movie2title.json'), mode='wb') as f:
   pickle.dump(movie2title, f)
