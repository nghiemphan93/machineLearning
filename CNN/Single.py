from sklearn import preprocessing
from keras.utils import to_categorical
from keras import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import pandas as pd
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/fashionmnist/fashion-mnist_train.csv")
data = data.values

trainData    = data[:, 1:]
trainLabel   = data[:, 0:1]
trainData = preprocessing.normalize(trainData)
trainData = np.reshape(trainData, (60000, 28, 28, 1))
trainLabel = to_categorical(trainLabel)


# Train model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(trainData, trainLabel, epochs=10, batch_size=128, validation_split=0.2)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
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