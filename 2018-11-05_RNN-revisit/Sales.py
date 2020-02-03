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
# from keras.models import Sequential
# from keras.layers import Dropout, CuDNNLSTM, LSTM
# from keras.layers.core import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from tensorflow.keras.models import Sequential

columns = ["adjustments", "unadjusted", "seasonallyAdjusted"]
df = pd.read_csv("./canSales.csv",
                 skiprows=7,
                 skipfooter=9,
                 engine="python",
                 names=columns)

df["adjustments"] = pd.to_datetime(df["adjustments"]) + MonthEnd(1)
df = df.set_index("adjustments")

# Split data
splitDate = pd.Timestamp("01-01-2011")
trainData = df.loc[:splitDate, ["unadjusted"]]
testData: DataFrame = df.loc[splitDate:, ["unadjusted"]]

'''
ax = trainData.plot()
testData.plot(ax=ax)
plt.legend(["train", "test"])
plt.show()
'''

sc = MinMaxScaler()
trainSC = sc.fit_transform(trainData)
testSC = sc.transform(testData)

# Split train data and test data
xTrain = trainSC[:-1]
yTrain = trainSC[1:]

xTest = testSC[:-1]
yTest = testSC[1:]

'''
# Fully connected model
model = Sequential()
model.add(Dense(12, input_shape=(xTrain.shape[1],), activation="relu"))
model.add(Dense(1))
model.compile(loss="mse",
              optimizer="adam")
model.summary()

earlyStop = EarlyStopping(monitor="loss",
                          patience=1,
                          verbose=1)
model.fit(xTrain, yTrain,
          epochs=200,
          batch_size=2,
          verbose=1,
          callbacks=[earlyStop])
yPredicted = model.predict(xTest)

base = np.asarray((range(len(yTest))))
plt.plot(base, yTest)
plt.plot(base, yPredicted)
plt.show()
print(yTest)
'''

# Recurrent Model
TIMESTEP = 1
xTrain = xTrain.reshape((xTrain.shape[0], TIMESTEP, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], TIMESTEP, xTest.shape[1]))
print(xTrain.shape)
print(xTest.shape)

model = Sequential()
model.add(CuDNNLSTM(50, input_shape=(xTrain.shape[1], xTrain.shape[2]),
               return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mse",
              optimizer="adam")
earlyStop = EarlyStopping(monitor="loss",
                          patience=1,
                          verbose=1)
model.fit(xTrain, yTrain,
          epochs=30,
          verbose=1,
          batch_size=1,
          callbacks=[])

yPredicted = model.predict(xTest)

base = np.asarray((range(len(yTest))))
plt.plot(base, yTest)
plt.plot(base, yPredicted)
plt.show()
