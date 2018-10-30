import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, CuDNNLSTM, Bidirectional, GRU, CuDNNGRU, SpatialDropout1D, Dropout, Conv2D, Conv1D, GlobalMaxPooling1D
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
EMBED_DIM        = 100
NUMB_FILTERS     = 256


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

embeddingMatrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
for word, i in wordIndex.items():
   if i < VOCAB_SIZE:
      embeddingVector = embeddingIndex.get(word)
      if embeddingVector is not None:
         embeddingMatrix[i] = embeddingVector

print(embeddingMatrix)

# Train model
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE,
                    output_dim=EMBED_DIM,
                    input_length=MAX_COMMENT_SIZE,
                    weights=[embeddingMatrix],
                    trainable=False))

model.add(Conv1D(filters=NUMB_FILTERS,
                 kernel_size=3,
                 activation="relu"))
model.add(SpatialDropout1D(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation="sigmoid"))


'''
model.add(Flatten())
model.add((Dense(32, activation="relu")))
model.add(Dense(1, activation="sigmoid"))
model.layers[0].set_weights([embeddingMatrix])
model.layers[0].trainable = True
'''
model.summary()


model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["acc"])
history = model.fit(data, label,
                    batch_size=256,
                    epochs=20,
                    validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()