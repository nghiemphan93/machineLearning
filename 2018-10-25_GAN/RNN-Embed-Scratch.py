import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Bidirectional, Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU, SpatialDropout1D, Dropout, Conv2D, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')

trainPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/sentimentClassification/training.txt"
testPath = "C:/Users/phan/OneDrive - adesso Group/DataSet/sentimentClassification/testdata.txt"

def loadData():
   file  = open(trainPath, encoding="utf8")
   line  = file.readline()
   data  = []
   label = []

   while(line != ""):
      temp  = line.strip().split(sep="\t")
      data.append(temp[1].strip("\n"))
      label.append(temp[0])

      line = file.readline()
   return data, label

def calMaxLen(data):
   max = 0
   for i in range(len(data)):
      if len(data[i]) > max:
         max = len(data[i])
   return max

text, label = loadData()

MAX_COMMENT_SIZE = calMaxLen(text)
VOCAB_SIZE       = 10000     # consider top 10000 words in dictionary
EMBED_SIZE       = 100
NUMB_FILTERS     = 32


tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(text)

sequence = tokenizer.texts_to_sequences(text)
wordIndex = tokenizer.word_index

data  = pad_sequences(sequence, maxlen=MAX_COMMENT_SIZE)
label = np.asarray(label)

# Shuffle data and labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data  = data[indices]
label = label[indices]

model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE,
                    output_dim=EMBED_SIZE,
                    input_length=MAX_COMMENT_SIZE))
model.add(SpatialDropout1D(0.2))
#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(CuDNNLSTM(64))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["acc"])
history = model.fit(data, label,
                    batch_size=256,
                    epochs=10,
                    validation_split=0.2)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()