from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
#style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, LSTM
from keras.callbacks import EarlyStopping

# Import data
fileName = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/weather/jena_climate_2009_2016.csv"
df = pd.read_csv(fileName)
trainingSet = df.iloc[:, 1:].values

# Feature Scaling
sc = MinMaxScaler()
trainingSetScaled = sc.fit_transform(trainingSet)

# Split train and test data
TIME_STEPS = 4320        # Look back 3 months,
MULTI = int(len(trainingSetScaled) / TIME_STEPS)
MAX_LEN = MULTI * TIME_STEPS

trainingSetScaled = trainingSetScaled[:MAX_LEN, :]

trainData = trainingSetScaled[:]