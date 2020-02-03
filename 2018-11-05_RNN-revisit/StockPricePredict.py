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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
#np.set_printoptions(threshold=np.nan)

# Import Data
dfTrain = pd.read_csv("Google_Stock_Price_Train.csv")
trainingSet = dfTrain.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)


# Import test data
dfTest = pd.read_csv("Google_Stock_Price_Test.csv")
testSet = dfTest.iloc[:, 1:2].values

dataSetTotal = pd.concat((dfTrain["Open"], dfTest["Open"]), axis=0)

inputs = dataSetTotal[len(dataSetTotal)-len(testSet) - 60:].values
inputs = np.reshape(inputs, newshape=(len(inputs), 1))

inputs = sc.transform(inputs)

TIME_STEPS = 60
xTest = []
for i in range(TIME_STEPS, 80):
   xTest.append(inputs[i-TIME_STEPS:i, 0])
xTest = np.asarray(xTest)

xTest = np.reshape(xTest,
                   (xTest.shape[0], xTest.shape[1], 1))


model: Sequential = load_model("stockPrice.h5")

predictedPrices = model.predict(xTest)

predictedPrices = sc.inverse_transform(predictedPrices)

base = np.asarray(range(20))
plt.plot(base, testSet, color="red", label="Real Price")
plt.plot(base, predictedPrices, color="blue", label="Predicted Price")
plt.title("Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price in $")
plt.legend()
plt.show()