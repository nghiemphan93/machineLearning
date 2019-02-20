import numpy as np
import keras
from keras.models import Sequential, Model
import os
from keras.layers import Input, Dense, Conv2D, Deconv2D, BatchNormalization
from keras.layers import Dropout, Flatten, Activation, Reshape, Conv2DTranspose
from keras.layers import UpSampling2D
from keras.optimizers import RMSprop

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

<<<<<<< HEAD
dataName       = "dragon"
dataPathLaptop = "C:/Users/phan/OneDrive - adesso Group/DataSet/" + dataName + ".npy"
dataPath       = "D:/OneDrive - adesso Group/DataSet/" + dataName + ".npy"
data = np.load(dataPathLaptop)
=======
dataName = "eye"
dataPath = "D:/OneDrive - adesso Group/DataSet/" + dataName + ".npy"
data = np.load(dataPath)
>>>>>>> master

data = data/255
data = np.reshape(data, (data.shape[0], 28, 28, 1))
imgWidth, imgHeight = data.shape[1:3]

<<<<<<< HEAD
print(data.shape)

'''
OFFSET = 200
=======
OFFSET = 500
>>>>>>> master

for index in range(10):
   plt.figure(figsize=(5, 5))
   for k in range(16):
      plt.subplot(4, 4, k+1)
      plt.imshow(data[k + OFFSET*index, :, :, 0], cmap="gray")
      plt.axis("off")
   plt.tight_layout()
   fileName = dataName + "-truth-" + str(index) + ".png"
   plt.savefig(fname=fileName)
   plt.show()
'''