import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, SimpleRNN, CuDNNLSTM, CuDNNGRU
from keras import Sequential
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import os

maxFeatures = 10000
maxLen = 500
batchSize = 32
VOCAB_SIZE       = 10000     # consider top 10000 words in dictionary
EMBED_DIM        = 100

print("loading...")
(inputTrain, yTrain), (inputTest, yTest) = imdb.load_data(num_words=maxFeatures)



inputTrain = sequence.pad_sequences(inputTrain, maxlen=maxLen)
inputTest = sequence.pad_sequences(inputTest, maxlen=maxLen)


# Process embedding
gloveFolder = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/glove6b"
gloveFile = "glove.6B.100d.txt"

embeddingIndex = {}
f = open(os.path.join(gloveFolder, gloveFile), encoding="utf8")
for line in f:
   values = line.split()
   word = values[0]
   coeffs = np.asarray(values[1:], dtype="float32")
   embeddingIndex[word] = coeffs
f.close()

embeddingMatrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
for word, i in wordIndex.items():
   if i < VOCAB_SIZE:
      embeddingVector = embeddingIndex.get(word)
      if embeddingVector is not None:
         embeddingMatrix[i] = embeddingVector



model = Sequential()
model.add(Embedding(maxFeatures, 32, weights=[embeddingMatrix], trainable=False))
model.add(CuDNNLSTM(32))
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
