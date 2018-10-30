import os, math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Bidirectional, Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU, SpatialDropout1D, Dropout, Conv2D, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.preprocessing import normalize
style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)



def loadData():
   trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/electricity/LD2011_2014.txt"
   df = pd.read_csv(trainPath, delimiter=";",  nrows=10000, decimal=",")
   df = df.rename(columns={"Unnamed: 0": "time"})
   df["time"] = pd.to_datetime(df["time"])
   df = df.set_index("time")



   kunde250 = df[["MT_250"]]
   kunde250["Kunde 250"] = kunde250["MT_250"]
   kunde250 = kunde250.drop("MT_250", axis=1)

   print(kunde250)
   print(kunde250.info())

   kunde250.iloc[384:576, :].plot()
   plt.show()


trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/electricity/LD2011_2014.txt"
df = pd.read_csv(trainPath, delimiter=";",   decimal=",")
df = df.rename(columns={"Unnamed: 0": "time"})
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")


kunde = df[["MT_250"]].copy()
kunde["training"] = kunde["MT_250"]
kunde = kunde.drop("MT_250", axis=1)
#kunde["target"] = kunde["training"].shift(periods=-1)
#kunde = kunde.dropna(how="any")

TIME_STEPS = 20

temp = kunde.values
temp = normalize(temp, axis=0)
data = temp[:, 0]
#data = np.reshape(data, (data.shape[0], 1))

x = np.zeros((data.shape[0], TIME_STEPS))
Y = np.zeros((data.shape[0], 1))

for i in range(len(data) - TIME_STEPS - 1):
   x[i] = data[i:i + TIME_STEPS].T
   Y[i] = data[i + TIME_STEPS + 1]


x = np.expand_dims(x, axis=2)


trainingData = x[:int(0.8*len(x))]
trainingLabel = Y[:int(0.8*len(Y))]
testData = x[int(0.8*len(x)):]
testLabel = Y[int(0.8*len(Y)):]



print(trainingData.shape)
print(testData.shape)
print(testLabel.shape)


'''
trainingData = np.reshape(trainingData, (trainingData.shape[0], trainingData.shape[1], 1))
testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1))
'''


model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(TIME_STEPS, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["mean_squared_error"])
model.fit(trainingData, trainingLabel, epochs=20, batch_size=96, validation_split=0.2, shuffle=False)

score, _ = model.evaluate(testData, testLabel, batch_size=96)

rmse = math.sqrt(score)
print(testData)
print(testLabel)
result = model.predict(testData)
print(result)
print(rmse)
print(score)

length = range(len(x))
plt.plot(length, data, "b", label="data")
plt.plot(length[int(0.8*len(length)):], result, "r", label="predict")
plt.legend()
plt.show()
