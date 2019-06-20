# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

sns.set_style('white')
pd.set_option('display.expand_frame_repr', False)
# endregion

columns = ['userId', 'itemId', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=columns)
movieTitles = pd.read_csv('Movie_Id_Titles', names=['itemId', 'title'])
df: pd.DataFrame = pd.merge(left=df, right=movieTitles, on='itemId')
df.loc[:, 'itemId'] = df['itemId'] - 1
print(df.head())

NUMB_USERS = df['userId'].nunique() + 1
NUMB_ITEMS = df['itemId'].nunique() + 1


def predictUserBased(ratings: np.ndarray, similarity: np.ndarray):
   meanUserRatings = ratings.mean(axis=1)
   ratingDeviations = (ratings - meanUserRatings.reshape((NUMB_USERS, 1)))
   return meanUserRatings.reshape((NUMB_USERS, 1)) + similarity.dot(ratingDeviations) / np.abs(similarity).sum(
      axis=1).reshape((NUMB_USERS, 1))


from sklearn.metrics import mean_squared_error


def rmse(prediction: np.ndarray, testData: np.ndarray):
   prediction = prediction[testData.nonzero()].flatten()
   testData = testData[testData.nonzero()].flatten()
   return np.sqrt(mean_squared_error(prediction, testData))


train, test = train_test_split(df, test_size=0.25)
trainData = np.zeros(shape=(NUMB_USERS, NUMB_ITEMS))
for line in train.itertuples():
   trainData[line[1], line[2]] = line[3]
print(trainData.head())

'''
nUsers = df['userId'].nunique()
nItems = df['itemId'].nunique()
trainData, testData = train_test_split(df, test_size=0.25)
trainDataMatrix = np.zeros(shape=(nUsers, nItems))
for line in trainData.itertuples():
    trainDataMatrix[line[1], line[2]-1] = line[3] 
    
testDataMatrix = np.zeros(shape=(nUsers, nItems))
for line in testData.itertuples():
    testDataMatrix[line[1], line[2]-1] = line[3]
    
userSimilarity = cosine_similarity(trainDataMatrix, metric='cosine')
itemSimilarity = cosine_similarity(trainDataMatrix.T, metric='cosine')

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
    
itemPrediction = predict_fast_simple(train, itemSimilarity, kind='item')
userPrediction = predict_fast_simple(train, userSimilarity, kind='user')

print( 'User-based CF MSE: ' + str(get_mse(userPrediction, testDataMatrix)))
print( 'Item-based CF MSE: ' + str(get_mse(itemPrediction, testDataMatrix)))
'''
