# region Import
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

plt.rcParams["axes.grid"] = False
sns.set()
# plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
from PIL import Image
import os


# endregion

# tf.enable_eager_execution()


def getFilePath():
   trainFolder = './data/train'
   testFolder = './data/test'
   outlinesFolder = './data/outlines'

   trainNames = os.listdir(trainFolder)
   outlinesNames = os.listdir(outlinesFolder)
   testNames = os.listdir(testFolder)
   trainNames.sort()
   outlinesNames.sort()
   testNames.sort()

   trainPath = []
   outlinesPath = []
   testPath = []

   for i in range(len(trainNames)):
      trainPath.append(os.path.join(trainFolder, trainNames[i]))
      outlinesPath.append(os.path.join(outlinesFolder, outlinesNames[i]))
      # print(trainPath[i], " : ", outlinesPath[i])

   for i in range(len(testNames)):
      testPath.append(os.path.join(testFolder, testNames[i]))
      # print(testPath[i])
   return trainPath, outlinesPath, testPath


def loadImage(imagePath, outlinePath):
   imageString = tf.read_file(imagePath)
   outlineString = tf.read_file(outlinePath)
   image = tf.image.decode_image(imageString, channels=3)
   outline = tf.image.decode_image(outlineString, channels=1)

   # Resize down to 224 x 224
   image = tf.image.resize_image_with_pad(image=image,
                                          target_height=224,
                                          target_width=224)
   outline = tf.image.resize_image_with_pad(image=outline,
                                            target_height=224,
                                            target_width=224)
   # expand batch dimension
   image = np.expand_dims(image, axis=0)
   outline = np.expand_dims(outline, axis=0)

   # scale down to 0-1
   image = image / 255
   outline = outline / 255

   return image, outline


def loadTestImage(testImagePath):
   imageString = tf.read_file(testImagePath)
   image = tf.image.decode_image(imageString, channels=3)
   image = tf.image.resize_image_with_pad(image=image,
                                          target_height=224,
                                          target_width=224)
   image = np.expand_dims(image, axis=0)
   image = image / 255
   return image


def loadTestData(xTest):
   numbImages = len(xTest)
   testImages = np.ndarray(shape=(numbImages, 224, 224, 3))

   for i in range(numbImages):
      testImages[i] = loadTestImage(xTest[i])
   with open('./testImages.pickle', 'wb') as wf:
      pickle.dump(testImages, wf)
   return testImages


def loadTestDataPickle():
   with open('./testImages.pickle', 'rb') as rf:
      xTest = pickle.load(rf)
   return xTest


def loadTrainingData(X, y):
   numbImages = len(X)
   trainingImages = np.ndarray(shape=(numbImages, 224, 224, 3))
   trainingOutlines = np.ndarray(shape=(numbImages, 224, 224, 1))

   for i in range(numbImages):
      trainingImages[i], trainingOutlines[i] = loadImage(X[i], y[i])

   with open('./trainingImages.pickle', 'wb') as wf:
      pickle.dump(trainingImages, wf)
   with open('./trainingOutlines.pickle', 'wb') as wf:
      pickle.dump(trainingOutlines, wf)
   return trainingImages, trainingOutlines


def loadTrainingDataPickle():
   with open('./trainingImages.pickle', 'rb') as rf:
      X = pickle.load(rf)
   with open('./trainingOutlines.pickle', 'rb') as rf:
      y = pickle.load(rf)
   return X, y


def createData():
   trainPath, outlinesPath, testPath = getFilePath()
   loadTrainingData(trainPath, outlinesPath)
   loadTestData(testPath)


def defineModel2():
   FILTER_DIM = 64
   DROPOUT_RATE = 0.4
   Z_DIM = 100
   imgWidth, imgHeight = xTrain.shape[1:3]

   # define inputs:
   inputs = tf.keras.layers.Input((imgWidth, imgHeight, 3))

   # Convolutional layers
   conv1 = tf.keras.layers.Conv2D(filters=FILTER_DIM * 1,
                                  kernel_size=5,
                                  strides=2,
                                  padding="same",
                                  activation="relu")(inputs)
   conv1 = tf.keras.layers.Dropout(DROPOUT_RATE)(conv1)

   conv2 = tf.keras.layers.Conv2D(filters=FILTER_DIM * 2,
                                  kernel_size=5,
                                  strides=2,
                                  padding="same",
                                  activation="relu")(conv1)
   conv2 = tf.keras.layers.Dropout(DROPOUT_RATE)(conv2)

   conv3 = tf.keras.layers.Conv2D(filters=FILTER_DIM * 3,
                                  kernel_size=5,
                                  strides=2,
                                  padding="same",
                                  activation="relu")(conv2)
   conv3 = tf.keras.layers.Dropout(DROPOUT_RATE)(conv3)

   conv4 = tf.keras.layers.Conv2D(filters=FILTER_DIM * 4,
                                  kernel_size=5,
                                  strides=1,
                                  padding="same",
                                  activation="relu")(conv3)
   conv4 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(DROPOUT_RATE)(conv4))

   # Output Layer
   output = tf.keras.layers.Dense(1, activation="sigmoid")(conv4)
   reshape = tf.keras.layers.Reshape((Z_DIM,))(output)

   '''============================================================================'''

   # define inputs
   # inputs = tf.keras.layers.Input((Z_DIM,))

   # first dense layer
   dense1 = tf.keras.layers.Dense(7 * 7 * FILTER_DIM)(reshape)
   dense1 = tf.keras.layers.BatchNormalization(momentum=0.9)(dense1)
   dense1 = tf.keras.layers.Activation(activation="relu")(dense1)
   dense1 = tf.keras.layers.Reshape((7, 7, FILTER_DIM))(dense1)
   dense1 = tf.keras.layers.Dropout(DROPOUT_RATE)(dense1)

   # Deconvolutional layers
   conv1 = tf.keras.layers.UpSampling2D()(dense1)
   conv1 = tf.keras.layers.Conv2DTranspose(filters=int(FILTER_DIM / 2),
                                           kernel_size=5,
                                           strides=2,
                                           padding="same",
                                           activation=None)(conv1)
   conv1 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv1)
   conv1 = tf.keras.layers.Activation("relu")(conv1)

   conv2 = tf.keras.layers.UpSampling2D()(conv1)
   conv2 = tf.keras.layers.Conv2DTranspose(filters=int(FILTER_DIM / 4),
                                           kernel_size=5,
                                           strides=2,
                                           padding="same",
                                           activation=None)(conv2)
   conv2 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2)
   conv2 = tf.keras.layers.Activation("relu")(conv2)

   conv3 = tf.keras.layers.Conv2DTranspose(filters=int(FILTER_DIM / 8),
                                           kernel_size=5,
                                           strides=2,
                                           padding="same",
                                           activation=None)(conv2)
   conv3 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv3)
   conv3 = tf.keras.layers.Activation("relu")(conv3)

   # Output layer
   output = tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid")(conv3)

   # Model definition
   model = tf.keras.models.Model(inputs=inputs, outputs=output)
   model.summary()

   model.compile(loss="binary_crossentropy",
                 optimizer=tf.keras.optimizers.RMSprop(lr=0.0004,
                                                       decay=3e-8,
                                                       clipvalue=1.0))

   model.fit(xTrain, yTrain,
             epochs=50,
             batch_size=128,
             validation_split=0.1)
   return model


def defineModel3():
   FILTER_DIM = 4
   DROPOUT_RATE = 0.4
   Z_DIM = 100
   imgWidth, imgHeight = xTrain.shape[1:3]

   # define inputs:
   inputs = keras.layers.Input((imgWidth, imgHeight, 3))

   # Convolutional layers
   conv1 = keras.layers.Conv2D(filters=FILTER_DIM * 4,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               activation="relu")(inputs)
   conv1 = keras.layers.Dropout(DROPOUT_RATE)(conv1)

   conv2 = keras.layers.Conv2D(filters=FILTER_DIM * 3,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               activation="relu")(conv1)
   conv2 = keras.layers.Dropout(DROPOUT_RATE)(conv2)

   conv3 = keras.layers.Conv2D(filters=FILTER_DIM * 2,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               activation="relu")(conv2)
   conv3 = keras.layers.Dropout(DROPOUT_RATE)(conv3)

   conv4 = keras.layers.Conv2D(filters=FILTER_DIM * 1,
                               kernel_size=5,
                               strides=1,
                               padding="same",
                               activation="relu")(conv3)
   conv4 = keras.layers.Flatten()(keras.layers.Dropout(DROPOUT_RATE)(conv4))

   # Output Layer
   # output = keras.layers.Dense(1, activation="sigmoid")(conv4)
   # reshape = keras.layers.Reshape((Z_DIM,))(conv4)

   '''============================================================================'''

   # define inputs
   # inputs = tf.keras.layers.Input((Z_DIM,))

   # first dense layer
   dense1 = keras.layers.Dense(7 * 7 * FILTER_DIM)(conv4)
   dense1 = keras.layers.BatchNormalization(momentum=0.9)(dense1)
   dense1 = keras.layers.Activation(activation="relu")(dense1)
   dense1 = keras.layers.Reshape((7, 7, FILTER_DIM))(dense1)
   dense1 = keras.layers.Dropout(DROPOUT_RATE)(dense1)

   # Deconvolutional layers
   conv1 = keras.layers.UpSampling2D()(dense1)
   conv1 = keras.layers.Conv2DTranspose(filters=int(FILTER_DIM * 1),
                                        kernel_size=5,
                                        strides=2,
                                        padding="same",
                                        activation=None)(conv1)
   conv1 = keras.layers.BatchNormalization(momentum=0.9)(conv1)
   conv1 = keras.layers.Activation("relu")(conv1)

   conv2 = keras.layers.UpSampling2D()(conv1)
   conv2 = keras.layers.Conv2DTranspose(filters=int(FILTER_DIM * 2),
                                        kernel_size=5,
                                        strides=2,
                                        padding="same",
                                        activation=None)(conv2)
   conv2 = keras.layers.BatchNormalization(momentum=0.9)(conv2)
   conv2 = keras.layers.Activation("relu")(conv2)

   conv3 = keras.layers.Conv2DTranspose(filters=int(FILTER_DIM * 4),
                                        kernel_size=5,
                                        strides=2,
                                        padding="same",
                                        activation=None)(conv2)
   conv3 = keras.layers.BatchNormalization(momentum=0.9)(conv3)
   conv3 = keras.layers.Activation("relu")(conv3)

   # Output layer
   output = keras.layers.Conv2D(filters=1,
                                kernel_size=5,
                                padding="same",
                                activation="sigmoid")(conv3)

   # Model definition
   model = keras.models.Model(inputs=inputs, outputs=output)
   model.summary()

   model.compile(loss="binary_crossentropy",
                 optimizer=keras.optimizers.RMSprop(lr=0.0004,
                                                    decay=3e-8,
                                                    clipvalue=1.0))

   model.fit(xTrain, yTrain,
             epochs=50,
             batch_size=128,
             validation_split=0.1)
   return model


def defineModel():
   FILTER_DIM = 64
   autoencoder = keras.models.Sequential()

   # Encoder Layers
   autoencoder.add(keras.layers.Conv2D(FILTER_DIM * 2, (2, 2),
                                       activation='relu',
                                       padding='same',
                                       input_shape=xTrain.shape[1:]))
   autoencoder.add(keras.layers.MaxPooling2D((2, 2),
                                             padding='same'))

   # Flatten encoding for visualization
   autoencoder.add(keras.layers.Flatten())
   autoencoder.add(keras.layers.Reshape((112, 112, FILTER_DIM * 2)))

   # Decoder Layers
   autoencoder.add(keras.layers.Conv2D(FILTER_DIM * 2, (2, 2),
                                       activation='relu',
                                       padding='same'))
   autoencoder.add(keras.layers.UpSampling2D((2, 2)))

   autoencoder.add(keras.layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same'))
   # autoencoder.add(tf.keras.layers.Reshape((224, 224, 1)))
   autoencoder.summary()

   autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
   autoencoder.fit(xTrain, yTrain,
                   epochs=20,
                   batch_size=8)
   return autoencoder


xTrain, yTrain = loadTrainingDataPickle()
xTest = loadTestDataPickle()
# yTrain = np.where(yTrain <= 0, yTrain, 1)
model = defineModel()

predicted = model.predict(xTest)
for i in range(len(predicted)):
   plt.rcParams["axes.grid"] = False
   plt.figure(figsize=(4, 8))
   fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
   ax[0].imshow(predicted[i, :, :, 0], cmap="gray")
   ax[1].imshow(xTest[i])
   plt.axis("off")
   plt.tight_layout()
   fileName = "./result/" + str(i) + ".png"
   fig.savefig(fname=fileName)

'''
predicted = model.predict(xTest)

temp = predicted[50].reshape((224, 224))
image = xTrain[20]
test = xTest[20]
label = yTrain[20].reshape((224, 224))
plt.rcParams["axes.grid"] = False
plt.imshow(temp, cmap='gray')
plt.show()


'''
