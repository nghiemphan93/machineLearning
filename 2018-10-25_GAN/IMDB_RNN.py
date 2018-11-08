import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, SimpleRNN, CuDNNLSTM, CuDNNGRU, LSTM
from keras import Sequential
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

def calMaxLen(data):
   max = 0
   for i in range(len(data)):
      if len(data[i]) > max:
         max = len(data[i])
   return max

maxFeatures = 10000
maxLen = 600
batchSize = 32

print("loading...")
(inputTrain, yTrain), (inputTest, yTest) = imdb.load_data(num_words=maxFeatures)

maxLen = calMaxLen(inputTrain)

inputTrain = sequence.pad_sequences(inputTrain, maxlen=maxLen)
inputTest = sequence.pad_sequences(inputTest, maxlen=maxLen)

print(inputTrain)


model = Sequential()
model.add(Embedding(input_dim=maxFeatures,
                    output_dim=32,
                    input_length=maxLen))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["acc"])
history = model.fit(inputTrain, yTrain,
                    epochs=10,
                    batch_size=batchSize,
                    validation_split=0.2)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
