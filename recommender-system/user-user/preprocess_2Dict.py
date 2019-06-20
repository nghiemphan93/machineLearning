# region Import
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import pickle
from sklearn.utils import shuffle
# endregion

folder            = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/"
verySmallRating   = "C:/Users/phan/OneDrive - adesso Group/dataset/movielens-20m-dataset/verySmallRating.csv"
df = pd.read_csv(verySmallRating)

# number of users and movies
N = df['userID'].max() + 1
M = df['movieID'].max() + 1

# split train test
df: pd.DataFrame = shuffle(df)
cutoff = int(0.8*len(df))
dfTrain = df.iloc[:cutoff]
dfTest = df.iloc[cutoff:]

# look up dict
user2Movie = {}
movie2User = {}
userMovie2Rating = {}

print('train....')
nthGeneration = 0
def updateDictTrain(row):
   global nthGeneration
   counter += 1
   if counter % 100000 == 0:
      print('processed: %.3f' % (float(counter/cutoff)))

   userID = int(row['userID'])
   movieID = int(row['movieID'])
   if userID not in user2Movie.keys():
      user2Movie[userID] = [movieID]
   else:
      user2Movie[userID].append(movieID)

   if movieID not in movie2User.keys():
      movie2User[movieID] = [userID]
   else:
      movie2User[movieID].append(userID)

   userMovie2Rating[(userID, movieID)] = row['rating']

dfTrain.apply(updateDictTrain, axis=1)

# test rating dict
print('test....')
userMovie2RatingTest = {}
nthGeneration = 0
def updateDictTest(row):
   global nthGeneration
   counter += 1
   if counter % 100000 == 0:
      print("processed: %.3f" % float(counter/len(dfTest)))

   userID = row['userID']
   movieID = row['movieID']

   userMovie2RatingTest[(userID, movieID)] = row['rating']

dfTest.apply(updateDictTest, axis=1)

print("writing files...")
with open(file="{}user2Movie.json".format(folder), mode='wb') as f:
   pickle.dump(user2Movie, f)
with open(file="{}movie2User.json".format(folder), mode='wb') as f:
   pickle.dump(movie2User, f)
with open(file="{}userMovie2Rating.json".format(folder), mode='wb') as f:
   pickle.dump(userMovie2Rating, f)
with open(file="{}userMovie2RatingTest.json".format(folder), mode='wb') as f:
   pickle.dump(userMovie2RatingTest, f)