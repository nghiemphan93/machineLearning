from sklearn import preprocessing
from keras.utils import to_categorical
from keras import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import pandas as pd
import numpy as np
from keras.callbacks import TensorBoard

# Load data
data = pd.read_csv("C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/fashionmnist/fashion-mnist_train.csv")
data = data.values

trainData    = data[:, 1:]
trainLabel   = data[:, 0:1]
trainData = preprocessing.normalize(trainData)
trainData = np.reshape(trainData, (60000, 28, 28, 1))
trainLabel = to_categorical(trainLabel)


# Train model
NUMB_NODES    = [50, 100, 150, 200, 250]
NUMB_HIDDEN   = [1, 2, 3, 4, 5]


model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()

NAME = "logs/{}-nodes_{}-hidden".format(NODE, HIDDEN)
tensorboard = TensorBoard(log_dir=NAME)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(trainData, trainLabel, epochs=10, batch_size=256, callbacks=[tensorboard])