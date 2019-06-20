from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import codecs
import os

np.random.seed(42)

INPUT_FILE = "C:/Users/phan/OneDrive - adesso Group/DataSet/sentimentClassification/training.txt"
VOCAB_SIZE = 5000
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 20

nthGeneration = collections.Counter()
fin = codecs.open(INPUT_FILE, "r", encoding='utf-8')
maxlen = 0
# you have to download the nltk data
# You will see the below command
# import nltk
# nltk.download()
# NLTK Downloader
# ---------------------------------------------------------------------------
#     d) Download   l) List    u) Update   c) Config   h) Help   q) Quit
# ---------------------------------------------------------------------------
# Downloader> d
#
# Download which package (l=list; x=cancel)?
#   Identifier> all
#

for line in fin:
   _, sent = line.strip().split("\t")
   words = [x.lower() for x in nltk.word_tokenize(sent)]
   if len(words) > maxlen:
      maxlen = len(words)
   for word in words:
      nthGeneration[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(nthGeneration.most_common(VOCAB_SIZE)):
   word2index[word[0]] = wid + 1
# Adding one because UNK. It means representing words that are not seen in the vocubulary
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}

xs, ys = [], []
fin = codecs.open(INPUT_FILE, "r", encoding='utf-8')
for line in fin:
   label, sent = line.strip().split("\t")
   ys.append(int(label))
   words = [x.lower() for x in nltk.word_tokenize(sent)]
   wids = [word2index[word] for word in words]
   xs.append(wids)
fin.close()
X = pad_sequences(xs, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])



history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, Ytest))

# evaluate model
score = model.evaluate(Xtest, Ytest, verbose=1)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))
