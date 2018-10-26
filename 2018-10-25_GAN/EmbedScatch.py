import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU, SpatialDropout1D, Dropout, Conv2D, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import pandas as pd

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
model.add(Conv1D(filters=NUMB_FILTERS,
                 kernel_size=3,
                 activation="relu"))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["acc"])
history = model.fit(data, label,
                    batch_size=512,
                    epochs=10,
                    validation_split=0.2)

