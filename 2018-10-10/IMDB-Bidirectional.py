import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU
import matplotlib.pyplot as plt

trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/imdb/train"
testPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/imdb/test"

labels   = []
texts    = []

for labelType in ["neg", "pos"]:
   folder = os.path.join(trainPath, labelType)
   for fileName in os.listdir(folder):
      if fileName[-4:] == ".txt":
         f = open(os.path.join(folder, fileName), encoding="utf8")
         texts.append(f.read())
         f.close()
         if labelType == "neg":
            labels.append(0)
         else:
            labels.append(1)

maxlen            = 600    # takes first 600 words
trainningSamples  = 10000    # train on 10000 samples
validationSamples = 2000  # validate on 2000 samples
maxWords          = 10000  # consider top 10000 words in dataset

tokenizer = Tokenizer(num_words=maxWords)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
wordIndex = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

# Shuffle data and labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Split data
trainData   = data[:trainningSamples]
trainLabel  = labels[:trainningSamples]
validData   = data[trainningSamples: trainningSamples + validationSamples]
validLabel  = labels[trainningSamples: trainningSamples + validationSamples]


# Process embedding
gloveFolder = "C:/Users/phan/OneDrive - adesso Group/DataSet/glove6b"
gloveFile = "glove.6B.100d.txt"

embeddingIndex = {}
f = open(os.path.join(gloveFolder, gloveFile), encoding="utf8")
for line in f:
   values = line.split()
   word = values[0]
   coeffs = np.asarray(values[1:], dtype="float32")
   embeddingIndex[word] = coeffs
f.close()

embeddingDim = 100
embeddingMatrix = np.zeros((maxWords, embeddingDim))
for word, i in wordIndex.items():
   if i < maxWords:
      embeddingVector = embeddingIndex.get(word)
      if embeddingVector is not None:
         embeddingMatrix[i] = embeddingVector


# Train model
model = Sequential()
model.add(Embedding(maxWords, 32))
model.add(Bidirectional(CuDNNLSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(trainData, trainLabel, epochs=10, batch_size=128, validation_split=0.2)


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