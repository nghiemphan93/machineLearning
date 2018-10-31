import cv2
from CustomCallbacks import BatchedTensorBoard
import shutil
from time import time
import keras
import os
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
import random
from keras.preprocessing import image
from matplotlib import style
#style.use('fivethirtyeight')

path = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/number/original"
whereTo = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/number/presentation"

def loadData():
   if os.path.exists(whereTo):
      shutil.rmtree(whereTo)
   os.mkdir(whereTo)

   for folderName in os.listdir(path):
      subFolder = os.path.join(path, folderName)
      listFiles = os.listdir(subFolder)

      randomIndex = random.randint(0, len(listFiles))
      randomFileName = listFiles[randomIndex]

      print(randomFileName)

      src = os.path.join(subFolder, randomFileName)
      dst = os.path.join(whereTo, randomFileName)
      shutil.copyfile(src, dst)

'''
while True:
   number = random.randint(0, 180)
   print(number)
'''

'''
listFiles = os.listdir(whereTo)
images = []
for fileName in listFiles:
   img = image.load_img(os.path.join(whereTo, fileName))
   images.append(img)

print(images[7])
w=10
h=10
fig=plt.figure(figsize=(50, 50))
columns = 10
rows = 5
counter = 0
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[counter])
    counter = counter + 1
plt.show()
'''