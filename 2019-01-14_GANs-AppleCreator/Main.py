import numpy as np
import keras
from keras.models import Sequential, Model
import os
from keras.layers import Input, Dense, Conv2D, Deconv2D, BatchNormalization
from keras.layers import Dropout, Flatten, Activation, Reshape, Conv2DTranspose
from keras.layers import UpSampling2D
from keras.optimizers import RMSprop
from PIL import Image
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

dataName = "dragon"
dataPath = "D:/OneDrive - adesso Group/DataSet/" + dataName + ".npy"
data = np.load(dataPath)

data = data/255
data = np.reshape(data, (data.shape[0], 28, 28, 1))
imgWidth, imgHeight = data.shape[1:3]

FILTER_DIM    = 64
DROPOUT_RATE   = 0.4

def discriminatorBuilder():
   # define inputs:
   inputs = Input((imgWidth, imgHeight, 1))

   # Convolutional layers
   conv1 = Conv2D(filters=FILTER_DIM * 1,
                  kernel_size=5,
                  strides=2,
                  padding="same",
                  activation="relu")(inputs)
   conv1 = Dropout(DROPOUT_RATE)(conv1)

   conv2 = Conv2D(filters=FILTER_DIM * 2,
                  kernel_size=5,
                  strides=2,
                  padding="same",
                  activation="relu")(conv1)
   conv2 = Dropout(DROPOUT_RATE)(conv2)

   conv3 = Conv2D(filters=FILTER_DIM * 4,
                  kernel_size=5,
                  strides=2,
                  padding="same",
                  activation="relu")(conv2)
   conv3 = Dropout(DROPOUT_RATE)(conv3)

   conv4 = Conv2D(filters=FILTER_DIM * 8,
                  kernel_size=5,
                  strides=1,
                  padding="same",
                  activation="relu")(conv3)
   conv4 = Flatten()(Dropout(DROPOUT_RATE)(conv4))

   # Output Layer
   output = Dense(1, activation="sigmoid")(conv4)

   # Model definition
   model = Model(inputs=inputs, outputs=output)
   model.summary()

   return model

discriminator = discriminatorBuilder()
discriminator.compile(loss="binary_crossentropy",
                      optimizer=RMSprop(lr=0.0008,
                                        decay=6e-8,
                                        clipvalue=1.0),
                      metrics=["acc"])
Z_DIM       = 100
def generatorBuilder():
   # define inputs
   inputs = Input((Z_DIM,))

   # first dense layer
   dense1 = Dense(7 * 7 * FILTER_DIM)(inputs)
   dense1 = BatchNormalization(momentum=0.9)(dense1)
   dense1 = Activation(activation="relu")(dense1)
   dense1 = Reshape((7, 7, FILTER_DIM))(dense1)
   dense1 = Dropout(DROPOUT_RATE)(dense1)

   # Deconvolutional layers
   conv1 = UpSampling2D()(dense1)
   conv1 = Conv2DTranspose(filters=int(FILTER_DIM/2),
                           kernel_size=5,
                           padding="same",
                           activation=None)(conv1)
   conv1 = BatchNormalization(momentum=0.9)(conv1)
   conv1 = Activation("relu")(conv1)

   conv2 = UpSampling2D()(conv1)
   conv2 = Conv2DTranspose(filters=int(FILTER_DIM / 4),
                           kernel_size=5,
                           padding="same",
                           activation=None)(conv2)
   conv2 = BatchNormalization(momentum=0.9)(conv2)
   conv2 = Activation("relu")(conv2)

   conv3 = Conv2DTranspose(filters=int(FILTER_DIM / 8),
                           kernel_size=5,
                           padding="same",
                           activation=None)(conv2)
   conv3 = BatchNormalization(momentum=0.9)(conv3)
   conv3 = Activation("relu")(conv3)

   # Output layer
   output = Conv2D(filters=1,
                   kernel_size=5,
                   padding="same",
                   activation="sigmoid")(conv3)

   # Model definition
   model = Model(inputs=inputs, outputs=output)
   model.summary()

   return model

generator = generatorBuilder()

def combine_images(generated_images):
   generated_images = generated_images.reshape(generated_images.shape[0],
                                               generated_images.shape[3],
                                               generated_images.shape[1],
                                               generated_images.shape[2])
   num = generated_images.shape[0]
   width = int(math.sqrt(num))
   height = int(math.ceil(float(num) / width))
   shape = generated_images.shape[2:]
   image = np.zeros((height * shape[0], width * shape[1]),
                    dtype=generated_images.dtype)
   for index, img in enumerate(generated_images):
      i = int(index / width)
      j = index % width
      image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
         img[0, :, :]
   return image

# Create adversarial network
def adversarialBuilder(GEN, DISC):
   model = Sequential()
   model.add(GEN)
   model.add(DISC)

   model.compile(loss="binary_crossentropy",
                 optimizer=RMSprop(lr=0.0004,
                                   decay=3e-8,
                                   clipvalue=1.0),
                 metrics=["acc"])
   model.summary()

   return model

adversarialModel = adversarialBuilder(generator, discriminator)

# Train
def makeTrainable(model, isTrainable):
   model.trainable = isTrainable
   for layer in model.layers:
      layer.trainable = isTrainable

def train(epochs=100, batch=100):
   discMetrics    = []
   adverMetrics   = []

   runningDiscLoss   = 0
   runningDiscAcc    = 0
   runningAdverLoss  = 0
   runningAdverAcc   = 0

   for i in range(epochs):

      realImages = np.reshape(data[np.random.choice(data.shape[0], batch, replace=False)], (batch, 28, 28, 1))
      fakeImages = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))

      x = np.concatenate((realImages, fakeImages))
      y = np.ones([2*batch, 1])

      y[batch:, :] = 0

      makeTrainable(discriminator, True)

      discMetrics.append(discriminator.train_on_batch(x, y))
      runningDiscLoss   += discMetrics[-1][0]
      runningDiscAcc    += discMetrics[-1][1]

      makeTrainable(discriminator, False)

      noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
      y = np.ones([batch, 1])

      adverMetrics.append(adversarialModel.train_on_batch(noise, y))
      runningAdverLoss += adverMetrics[-1][0]
      runningAdverAcc += adverMetrics[-1][1]

      if i % 500 == 0:
         print('Epoch #{}'.format(i + 1))
         log_mesg = "%d: [D loss: %f, acc: %f]" % (i, runningDiscLoss / (i+1), runningDiscAcc / (i+1))
         log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, runningAdverLoss / (i+1), runningAdverAcc / (i+1))
         print(log_mesg)

         noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
         genImages = generator.predict(noise)


         plt.figure(figsize=(5, 5))
         for k in range(genImages.shape[0]):
            plt.subplot(4, 4, k+1)
            plt.imshow(genImages[k, :, :, 0], cmap="gray")
            plt.axis("off")
         plt.tight_layout()
         fileName = dataName + "-" + "epoch-" + str(i) + ".png"
         plt.savefig(fname=fileName)
         plt.show()


         '''
         image = combine_images(genImages)
         image = image * 127.5 + 127.5
         Image.fromarray(image.astype(np.uint8)).save(
            dataName + "-" + "epoch-" + str(i) + ".png")
         '''
   return adverMetrics, discMetrics

a_metrics_complete, d_metrics_complete = train(epochs=15000)
ax = pd.DataFrame(
    {
        'Generator': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }
).plot(title='Training Loss', logy=True)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
plt.show()

ax = pd.DataFrame(
    {
        'Generator': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }
).plot(title='Training Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
plt.show()