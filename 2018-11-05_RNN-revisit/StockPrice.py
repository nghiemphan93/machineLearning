from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import datetime
from matplotlib import style
style.use('fivethirtyeight')
from pandas.core.frame import DataFrame
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM
from keras.callbacks import EarlyStopping

# Import Data
df = pd.read_csv("Google_Stock_Price_Train.csv")
trainingSet = df.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)

# Creating a ds with 60 TIMESTEPS and 1 output
TIME_STEPS = 60
xTrain = []
yTrain = []

for i in range(TIME_STEPS, len(trainingSetScaled)):
   xTrain.append(trainingSetScaled[i-TIME_STEPS:i, 0])
   yTrain.append(trainingSetScaled[i, 0])
xTrain = np.asarray(xTrain)
yTrain = np.asarray(yTrain)

xTrain = np.reshape(xTrain,
                    (xTrain.shape[0], xTrain.shape[1], 1))


# Build model
model = Sequential()
model.add(CuDNNLSTM(units=50,
                    input_shape=(xTrain.shape[1], xTrain.shape[2]),
                    return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units=50,
                    return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units=50,
                    return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units=50,
                    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam",
              loss="mse")
model.fit(xTrain, yTrain,
          epochs=100,
          batch_size=32,
          verbose=2)
model.save("stockprice.h5")

