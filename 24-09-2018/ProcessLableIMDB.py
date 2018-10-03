from keras.callbacks import TensorBoard
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Dropout
from time import time

def filterData():
    imdbFolder  = "C:/Users/phan/Downloads/DataSet/imdb"
    trainFolder = os.path.join(imdbFolder, "train")

    data    = []
    label   = []

    labelTypeList = ["neg", "pos"]
    for labelType in labelTypeList:
        workingFolder = os.path.join(trainFolder, labelType)
        for fileName in os.listdir(workingFolder):
            if fileName[-4:] == ".txt":
                file = open(os.path.join(workingFolder, fileName), encoding="utf8")
                data.append(file.read())
                file.close()
                if labelType == "neg":
                    label.append(0)
                else:
                    label.append(1)
    np.savez_compressed("dataIMDB.npz", data=data)
    np.savez_compressed("labelIMDB.npz", label=label)

def loadData():
    data = np.load("dataIMDB.npz")["data"]
    label = np.load("labelIMDB.npz")["label"]
    return data, label

def loadTrainDataAndLabel():
    trainData = np.load("trainDataIMDB.npz")["trainData"]
    trainLabel = np.load("trainLabelIMDB.npz")["trainLabel"]
    return trainData, trainLabel

def saveTrainingDataLabel():
    data, label = loadData()

    maxLen = 200
    trainingSamples = len(data)
    validationSamples = int(0.2 * len(data))
    maxWords = 10000

    tokenizer = Tokenizer(num_words=maxWords)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    wordIndex = tokenizer.word_index

    newData = pad_sequences(sequences, maxlen=maxLen)
    newLabel = np.asarray(label)

    indices = np.arange(newData.shape[0])
    print(indices)
    np.random.shuffle(indices)
    print(indices)

    newData = newData[indices]
    newLabel = newLabel[indices]

    np.savez_compressed("trainDataIMDB.npz", trainData=newData)
    np.savez_compressed("trainLabelIMDB.npz", trainLabel=newLabel)

#saveTrainingDataLabel()
trainData, trainLabel = loadTrainDataAndLabel()

model = Sequential()
model.add(Embedding(10000, 8, input_length=200))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["acc"])
model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
history = model.fit(trainData, trainLabel,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[tensorboard])