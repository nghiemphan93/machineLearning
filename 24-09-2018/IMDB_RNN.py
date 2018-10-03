import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, SimpleRNN, CuDNNLSTM, CuDNNGRU
from keras import Sequential
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

maxFeatures = 10000
maxLen = 500
batchSize = 32

print("loading...")
(inputTrain, yTrain), (inputTest, yTest) = imdb.load_data(num_words=maxFeatures)



inputTrain = sequence.pad_sequences(inputTrain, maxlen=maxLen)
inputTest = sequence.pad_sequences(inputTest, maxlen=maxLen)

model = Sequential()
model.add(Embedding(maxFeatures, 32))
model.add(CuDNNGRU(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])
history = model.fit(inputTrain, yTrain, epochs=5, batch_size=batchSize, validation_split=0.2)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()