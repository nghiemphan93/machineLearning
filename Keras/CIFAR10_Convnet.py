import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard



(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
HEIGHT  = 32
WIDTH   = 32
CHANNEL = 3
NUMB_LABEL  = 10
VERBOSE     = 1
BATCH_SIZE  = 128
NUMB_EPOCHS = 10

# Augmenting
print("Augmenting training set images...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen2 = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(xTrain)
print(xTrain.shape)

print(HEIGHT, WIDTH, CHANNEL)

# Categorize labels
yTrain  = np_utils.to_categorical(yTrain, 10)
yTest   = np_utils.to_categorical(yTest, 10)

# Normalize training data
xTrain  = xTrain.astype("float32")
xTest   = xTest.astype("float32")
xTrain  = xTrain / 255
xTest   = xTest / 255

# Network
model   = Sequential()
model.add(Conv2D(64, (3, 3), padding="same",
                 input_shape=(HEIGHT, WIDTH, CHANNEL),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUMB_LABEL, activation="softmax"))
model.summary()

# Train


model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
tensorboard.set_model(model)
history = model.fit(xTrain, yTrain,
                    batch_size=BATCH_SIZE,
                    epochs=NUMB_EPOCHS,
                    validation_split=0.2,
                    verbose=VERBOSE,
                    callbacks=[])
score = model.evaluate(xTest, yTest, batch_size=BATCH_SIZE, verbose=VERBOSE)

print("Test score: ", score[0])
print("Test accuracy: ", score[1])


# Save model
'''
modelToJSON = model.to_json()
open("cifar10.json", "w").write(modelToJSON)
model.save_weights("cifar10_weights.h5", overwrite=True)
'''
model.save("CIFAR10_CONVNET.h5")