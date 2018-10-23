import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

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

maxlen            = 300    # takes first 100 words
trainningSamples  = 2000    # train on 200 samples
validationSamples = 10000  # validate on 10000 samples
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



model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))
# After the Embedding layer,
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(trainData, trainLabel,
                    epochs=40,
                    batch_size=32,
                    validation_split=0.2)
