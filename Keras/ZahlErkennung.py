from CustomCallbacks import BatchedTensorBoard
import shutil
from time import time
import keras
import cv2, os
import numpy as np
from keras import models, optimizers
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical

'''
def loadNumbers(folder: str):
    ZEHN, ELF, ZWOELF, DREIZEHN, VIERZEHN = countNumbers(folder)

    zehn = np.zeros((ZEHN, 56, 56, 3), dtype="float16")
    elf = np.zeros((ELF, 56, 56, 3), dtype="float16")
    zwoelf = np.zeros((ZWOELF, 56, 56, 3), dtype="float16")
    dreizehn = np.zeros((DREIZEHN, 56, 56, 3), dtype="float16")
    viewzehn = np.zeros((VIERZEHN, 56, 56, 1), dtype="float16")

    numbers  = np.array([np.zeros((numberCounters[0], 56, 56, 1), dtype="float16"),
                np.zeros((numberCounters[1], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[2], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[3], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[4], 56, 56, 3), dtype="float16")])

    numberIndex = [0,0,0,0,0]

    for filename in os.listdir(folder):
        if filename.startswith("10"):
            print(filename)
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            #img = image.load_img(os.path.join(folder, filename), target_size=(56, 56))
            #img = image.img_to_array(img)
            img = img / 255
            numbers[numberIndex[0]] = img
            numberIndex[0] += 1


    return numbers
'''

'''
def countNumbers(folder: str):
    ZEHN, ELF, ZWOELF, DREIZEHN, VIERZEHN = 0, 0, 0, 0, 0

    for filename in os.listdir(folder):
        if filename.startswith("10"):
            ZEHN += 1
        if filename.startswith("11"):
            ELF += 1
        if filename.startswith("12"):
            ZWOELF += 1
        if filename.startswith("13"):
            DREIZEHN += 1
        if filename.startswith("14"):
            VIERZEHN += 1
    return ZEHN, ELF, ZWOELF, DREIZEHN, VIERZEHN
'''

def countNumbers(folder: str):
    numberCounters = [0,0,0,0,0]
    # numberCounters[0], numberCounters[1], numberCounters[2], numberCounters[3], numberCounters[4]
    #  10                        11                  12                  13              14
    for filename in os.listdir(folder):
        if filename.startswith("10"):
            numberCounters[0] += 1
        if filename.startswith("11"):
            numberCounters[1] += 1
        if filename.startswith("12"):
            numberCounters[2] += 1
        if filename.startswith("13"):
            numberCounters[3] += 1
        if filename.startswith("14"):
            numberCounters[4] += 1
    return numberCounters



def loadNumbers(folder: str):
    numberCounters = countNumbers(folder)
    print(numberCounters)

    numbers  = np.array([np.zeros((numberCounters[0], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[1], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[2], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[3], 56, 56, 3), dtype="float16"),
                np.zeros((numberCounters[4], 56, 56, 3), dtype="float16")])

    numberIndex = [0,0,0,0,0]

    for filename in os.listdir(folder):
        if filename.startswith("10"):
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            img = img / 255
            numbers[0, numberIndex[0]] = img
            numberIndex[0] += 1
        if filename.startswith("11"):
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            img = img / 255
            numbers[1, numberIndex[1]] = img
            numberIndex[1] += 1
        if filename.startswith("12"):
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            img = img / 255
            numbers[2, numberIndex[2]] = img
            numberIndex[2] += 1
        if filename.startswith("13"):
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            img = img / 255
            numbers[3, numberIndex[3]] = img
            numberIndex[3] += 1
        if filename.startswith("14"):
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (56, 56)).astype(np.float16)])
            img = img / 255
            numbers[4, numberIndex[4]] = img
            numberIndex[4] += 1

    return numbers

def createTrainLabel():
    trainLabel = np.zeros((5*149, 1))

    zahl = 9
    for i in range(0, 149*5):
        if i % 149 == 0:
            zahl += 1
            print(zahl)

        trainLabel[i] = zahl
    return trainLabel



def getTrainData(folder: str):
    for filename in os.listdir(folder):
        if filename.startswith("10"):
         print(filename)
'''
trainFolder = "C:/Users/phan/Downloads/DataSet/zahl/train"
numbers = loadNumbers(trainFolder)

trainLabel = createTrainLabel()
print(trainLabel)
print(trainLabel.shape)



trainData = np.vstack([numbers[0],
                       numbers[1],
                       numbers[2],
                       numbers[3],
                       numbers[4]])
print(trainData.shape)


idx = np.random.permutation(len(trainData))
trainData, trainLabel = trainData[idx], trainLabel[idx]
print(trainLabel)
np.savez_compressed("steffanNumbers.npz", trainData=trainData, trainLabel=trainLabel)
'''

# Loading data
data = np.load("steffanNumbers.npz")
trainData = data["trainData"]
trainLabel = data["trainLabel"]
trainLabel = trainLabel - 10
trainLabel = to_categorical(trainLabel)
'''
for i in range(len(trainLabel)):
    print(trainLabel[i])
'''

# Create Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(56, 56, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(trainData, trainLabel, epochs=15, batch_size=64, validation_split=0.2)

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
model.save("steffanNumbers.h5")
