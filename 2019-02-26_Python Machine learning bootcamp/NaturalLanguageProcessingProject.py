import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, CuDNNLSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt


df = pd.read_csv("./data/yelp.csv")


df["length"] = df["text"].apply(len)

#sns.distplot(df["length"])
'''
stars = df["stars"].unique()
for star in stars:
    plt.figure(figsize=(12, 4))
    df[df["stars"] == star]["length"].plot.hist(label=star)
    plt.legend()
    plt.show()
'''
'''
g = sns.FacetGrid(df, col="stars")
g.map(plt.hist, "length")
plt.show()
'''

#print(df["stars"].value_counts())

# Cut out extra data
#data = df.loc[df["stars"] == 5].head(749)
data = pd.DataFrame()
for i in range(1, 6):
   data = pd.concat([data, df.loc[df["stars"] == i].head(749)], axis=0)

#data["length"].hist()
#plt.show()

### Train test
X: np.ndarray = data["text"].values
y: np.ndarray = data["stars"].values

maxlen            = 2000   # takes first 2000 words
trainningSamples  = 2996   # train on 2996 samples
validationSamples = 749    # validate on 749 samples
maxWords          = 10000  # consider top 10000 words in dataset
embeddingDim      = 100


### Tokenize words
tokenizer = Tokenizer(num_words=maxWords)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
wordIndex = tokenizer.word_index
indexWord = tokenizer.index_word
XTokenized = pad_sequences(sequences, maxlen=maxlen)

'''
print("X[0]: {}".format(X[0]))
print("sequences[0]: {}".format(sequences[0]))
print("XTokenized[0]: {}".format(XTokenized[0]))

print(len(X[0]))
print(len(sequences[0]))
print(len(XTokenized[0]))
print(type(wordIndex))
transformedMess = ""
transformedMess = [transformedMess.join(indexWord.get(index)) for index in sequences[0]]
print(transformedMess)
print("X[0]: {}".format(X[0]))
print(y[0])
'''

### Shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
XTokenized = XTokenized[indices]
y = y[indices]
y = y-1
y = to_categorical(y, num_classes=5)

### Split data
xTrain   = XTokenized[:trainningSamples]
xValid   = XTokenized[trainningSamples: trainningSamples + validationSamples]
yTrain   = y[:trainningSamples]
yValid   = y[trainningSamples: trainningSamples + validationSamples]



### Train model
model = Sequential()
model.add(Embedding(input_dim=maxWords,
                    output_dim=embeddingDim,
                    input_length=maxlen))
model.add(CuDNNLSTM(units=32, return_sequences=False))
model.add(Dense(units=5,
                activation="sigmoid"))
model.summary()

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["acc"])
history = model.fit(xTrain, yTrain,
                    epochs=20,
                    batch_size=32,
                    validation_data=(xValid, yValid))
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
