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
#style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize


def loadData():
   trainPath = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/electricity/LD2011_2014.txt"
   df = pd.read_csv(trainPath, delimiter=";", decimal=",")
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
df = pd.read_csv(trainPath, delimiter=";",  decimal=",")
df = df.rename(columns={"Unnamed: 0": "time"})
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")


kunde = df[["MT_250"]].copy()
kunde["training"] = kunde["MT_250"]
kunde = kunde.drop("MT_250", axis=1)
#kunde["target"] = kunde["training"].shift(periods=-1)
#kunde = kunde.dropna(how="any")
'''
kunde.iloc[1000:1500, -1].plot()
plt.title("Electricity Usage")
plt.show()
'''
print(kunde["training"])
LOOK_BACK = 672
PREDICT = 672

temp = kunde.values
#temp = normalize(temp, axis=0)
'''
temp = temp.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1), copy=False)
temp = scaler.fit_transform(temp)
'''
data = temp[:, 0]
#data = np.reshape(data, (data.shape[0], 1))


'''
x = np.zeros((data.shape[0], LOOK_BACK))
Y = np.zeros((data.shape[0], 1))

for i in range(len(data) - LOOK_BACK - 1):
   x[i] = data[i:i + LOOK_BACK].T
   Y[i] = data[i + LOOK_BACK + 1]

x = np.expand_dims(x, axis=2)
'''

# Test new
   

print(x.shape)

trainingData = x[:int(0.8*len(x))]
trainingLabel = Y[:int(0.8*len(Y))]
testData = x[int(0.8*len(x)):]
testLabel = Y[int(0.8*len(Y)):]




'''
trainingData = np.reshape(trainingData, (trainingData.shape[0], trainingData.shape[1], 1))
testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1))
'''


model = Sequential()
model.add(CuDNNGRU(64, input_shape=(LOOK_BACK, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNGRU(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNGRU(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss="mae",
              optimizer="adam",
              metrics=["mae"])
model.fit(trainingData, trainingLabel,
          epochs=10,
          batch_size=96,
          validation_split=0.2,
          shuffle=False)

mse, mae = model.evaluate(testData, testLabel, batch_size=96)



result = model.predict(testData)
#print(result)
print(mae)


length = range(len(x))
plt.title("Electricity Prediction")
plt.plot(length[-2000:-1700], data[-2000:-1700], "b", label="actual data")
plt.plot(length[-2000:-1700], result[-2000:-1700], "r", label="predict")
plt.legend()
plt.show()
