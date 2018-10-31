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
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')

def separateTrainValidation(baseFolder):
    originalFolder      = os.path.join(baseFolder, "original")
    samplesFolder       = os.path.join(baseFolder, "samples")
    sampleTrain         = os.path.join(samplesFolder, "train")
    sampleValidation    = os.path.join(samplesFolder, "validation")
    sampleTest          = os.path.join(samplesFolder, "test")
    if os.path.exists(samplesFolder):
        shutil.rmtree(samplesFolder)
    os.mkdir(samplesFolder)
    os.mkdir(sampleTrain)
    os.mkdir(sampleValidation)
    os.mkdir(sampleTest)

    shuffledIndices = np.random.permutation(180)

    for label in os.listdir(originalFolder):
        #print(subfolder, " ", len(os.listdir(os.path.join(originalFolder, subfolder))))
        fullSubFolder = os.path.join(originalFolder, label)
        filesTensor = np.asarray(os.listdir(fullSubFolder))
        filesTensor = filesTensor[shuffledIndices]

        # Copy 70% to train data
        sampleTrainLabel = os.path.join(sampleTrain, label)
        os.mkdir(sampleTrainLabel)
        for fileName in filesTensor[:126]:
            src = os.path.join(fullSubFolder, fileName)
            dst = os.path.join(sampleTrainLabel, fileName)
            shutil.copyfile(src, dst)



        # Copy 20% to validation data
        sampleValidationLabel = os.path.join(sampleValidation, label)
        os.mkdir(sampleValidationLabel)
        for fileName in filesTensor[126:162]:
            src = os.path.join(fullSubFolder, fileName)
            dst = os.path.join(sampleValidationLabel, fileName)
            shutil.copyfile(src, dst)

        # Copy 10% to testData
        sampleTestLabel = os.path.join(sampleTest, label)
        os.mkdir(sampleTestLabel)
        for fileName in filesTensor[162:]:
            src = os.path.join(fullSubFolder, fileName)
            dst = os.path.join(sampleTestLabel, fileName)
            shutil.copyfile(src, dst)



baseFolder = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/number"
#separateTrainValidation(baseFolder)
originalFolder = os.path.join(baseFolder, "original")
samplesFolder = os.path.join(baseFolder, "samples")
sampleTrain = os.path.join(samplesFolder, "train")
sampleValidation = os.path.join(samplesFolder, "validation")
sampleTest = os.path.join(samplesFolder, "test")

numbHiddenLayers  = [1, 2, 3]
numbUnitConv2d    = [32, 64, 128]
numbUniDense      = [128, 256, 512]

trainDatagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,)

trainBatches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    sampleTrain,
    target_size=(56, 56),
    batch_size=153,
    class_mode='categorical'
)
validBatches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    sampleValidation,
    target_size=(56, 56),
    batch_size=54,
    class_mode='categorical')


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

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(51, activation='softmax'))
model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])

history = model.fit_generator(trainBatches,
                              steps_per_epoch=42,
                              epochs=40,
                              validation_data=validBatches,
                              validation_steps=34)

#model.save("zahlErkennung3.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Number Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Number Training and validation loss')
plt.legend()
plt.show()