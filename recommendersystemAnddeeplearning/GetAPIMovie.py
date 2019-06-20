import tmdbsimple as tmdb
import urllib

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

columns = ['userId', 'itemId', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', names=columns, sep='\t')
movies = pd.read_csv('./ml-100k/u.item', sep='|', encoding='latin', names=range(24))
movies = movies.iloc[:, :2]
movies['year'] = movies.iloc[:, 1].str[-5:-1]
movies.iloc[:, 1] = movies.iloc[:, 1].str[:-6]
movies = movies.rename(columns={0: 'movieId', 1: 'title'})

print(movies.head())


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


print(getTopKMovieIndices(1))

showPoster(1)
