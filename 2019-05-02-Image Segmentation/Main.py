# region Import
import os
from typing import Tuple, List
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import tarfile
from urllib.request import urlretrieve

# endregion



def downloadFiles() -> None:
   """
   Download training, test and outline images

   :return: None
   """
   dataPath = os.path.join(os.getcwd(), 'data')
   if not os.path.exists(dataPath):
      os.mkdir(dataPath)
      IMAGE_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"

      print('Downloading images…')
      base_name   = os.path.basename(IMAGE_URL)
      path        = os.path.join('data', base_name)
      file_tmp    = urlretrieve(IMAGE_URL, filename=path)[0]
      print('Stored results at {}'.format(path))

      file_name, file_extension = os.path.splitext(base_name)
      tar = tarfile.open(file_tmp)
      tar.extractall('data')
      # os.remove(path)
      print('Extracted {}'.format(file_name))

      IMAGES_PATH = os.path.join('data', 'BSDS300', 'images', 'train')
      print('Found {} training images.'.format(len(os.listdir(IMAGES_PATH))))

      pairs = []
      image_paths = map(lambda x: os.path.join(IMAGES_PATH, x), os.listdir(IMAGES_PATH))

      # Create folder outlines for union combinations
      OUTLINES_PATH  = os.path.join('data')
      newpath        = OUTLINES_PATH + '//outlines'
      if not os.path.exists(newpath):
         os.makedirs(newpath)

      # Download union combination for each image
      print('Start downloading outlines...')

      for image_path in image_paths:
         OUTLINE_URL    = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench" \
                       "/BSDS300/html/images/human/normal/outline/color/union/" + os.path.basename(image_path)
         base_name      = os.path.basename(OUTLINE_URL)
         base_name_edit = base_name.replace(".jpg", "_union.jpg")
         # print('Downloading outline…' + base_name_edit)
         path           = os.path.join('data', 'outlines', base_name_edit)
         file_tmp       = urlretrieve(OUTLINE_URL, filename=path)[0]
         outline_path   = path

         pairs.append([image_path, outline_path])
      print('Download done!')


def getImagePaths() -> Tuple[List[str], List[str], List[str]]:
   """
   Get train, test and outline images paths and save to lists

   :return: trainPaths, outlinesPaths, testPaths
   """
   print(f'getting images paths...')
   trainFolder    = './data/BSDS300/images/train'
   testFolder     = './data/BSDS300/images/test'
   outlinesFolder = './data/outlines'

   trainNames     = os.listdir(trainFolder)
   outlinesNames  = os.listdir(outlinesFolder)
   testNames      = os.listdir(testFolder)

   trainNames.sort()
   outlinesNames.sort()
   testNames.sort()

   trainPaths     = []
   outlinesPaths  = []
   testPaths      = []

   for trainName, outlinesName in zip(trainNames, outlinesNames):
      trainPaths.append(os.path.join(trainFolder, trainName))
      outlinesPaths.append(os.path.join(outlinesFolder, outlinesName))

   for testName in testNames:
      testPaths.append(os.path.join(testFolder, testName))

   return trainPaths, outlinesPaths, testPaths


def readTrainImageAndOutline(imagePath: str, outlinePath: str) -> Tuple[np.ndarray, np.ndarray]:
   """
   Read a train image and an outline from path to ndarrays

   :param imagePath:
   :param outlinePath:
   :return: image, outline
   """
   imageString    = tf.read_file(imagePath)
   outlineString  = tf.read_file(outlinePath)
   image          = tf.image.decode_image(contents=imageString, channels=3)
   outline        = tf.image.decode_image(contents=outlineString, channels=1)

   # Resize down to 224 x 224
   image    = tf.image.resize_image_with_pad(image=image,
                                          target_height=224,
                                          target_width=224)
   outline  = tf.image.resize_image_with_pad(image=outline,
                                            target_height=224,
                                            target_width=224)
   # expand batch dimension
   image    = np.expand_dims(image, axis=0)
   outline  = np.expand_dims(outline, axis=0)

   # scale down to 0-1
   image    = image / 255
   outline  = outline / 255
   return image, outline


def readTestImage(testImagePath: str) -> np.ndarray:
   """
   Read a test image from path and save to ndarray

   :param testImagePath:
   :return: image
   """
   imageString = tf.read_file(testImagePath)
   image       = tf.image.decode_image(contents=imageString, channels=3)
   # Resize down to 224 x 224
   image       = tf.image.resize_image_with_pad(image=image,
                                          target_height=224,
                                          target_width=224)
   # expand batch dimension
   image       = np.expand_dims(image, axis=0)
   # scale down to 0-1
   image       = image / 255
   return image


def saveTrainData(trainImagePaths: List[str], trainOutlinePaths: List[str]) -> None:
   """
   Save train images and outlines to pickle file

   :param trainImagePaths:
   :param trainOutlinePaths:
   :return: None
   """
   print(f'saving train data...')
   numbImages     = len(trainImagePaths)
   trainImages    = np.ndarray(shape=(numbImages, 224, 224, 3))
   trainOutlines  = np.ndarray(shape=(numbImages, 224, 224, 1))

   for i in range(numbImages):
      trainImages[i], trainOutlines[i] = readTrainImageAndOutline(imagePath=trainImagePaths[i],
                                                                  outlinePath=trainOutlinePaths[i])

   with open(file='./trainImages.pickle', mode='wb') as wf:
      pickle.dump(trainImages, wf)
   with open(file='./trainOutlines.pickle', mode='wb') as wf:
      pickle.dump(trainOutlines, wf)


def loadTrainData() -> Tuple[np.ndarray, np.ndarray]:
   """
   Load train images and outlines from pickle file to ndarrays

   :return: trainImages, trainOutlines
   """
   print(f'loading train data...')
   with open(file='./trainImages.pickle', mode='rb') as rf:
      trainImages = pickle.load(rf)
   with open(file='./trainOutlines.pickle', mode='rb') as rf:
      trainOutlines = pickle.load(rf)
   return trainImages, trainOutlines


def saveTestData(testImagePaths: List[str]) -> None:
   """
   Save test images to pickle file

   :param testImagePaths: List[str]
   :return: None
   """
   print(f'saving test data...')
   numbImages = len(testImagePaths)
   testImages = np.ndarray(shape=(numbImages, 224, 224, 3))

   for i in range(numbImages):
      testImages[i] = readTestImage(testImagePath=testImagePaths[i])
   with open(file='./testImages.pickle', mode='wb') as wf:
      pickle.dump(testImages, wf)


def loadTestData() -> np.ndarray:
   """
   Load test images from pickle file to ndarray

   :return: testImages
   """
   print(f'loading test data')
   with open(file='./testImages.pickle', mode='rb') as rf:
      testImages: np.ndarray = pickle.load(rf)
   return testImages


def defineModel(xTrain: np.ndarray, yTrain: np.ndarray) -> tf.keras.models.Sequential:
   """
   Define the autoencoder model to train

   :param xTrain: np.ndarray
   :param yTrain: np.ndarray
   :return: autoencoder
   """
   print('defining model...')
   FILTER_DIM  = 50
   autoencoder = tf.keras.models.Sequential()

   # Encoder Layers
   autoencoder.add(tf.keras.layers.Conv2D(filters=FILTER_DIM * 2,
                                          kernel_size=(2, 2),
                                          activation='relu',
                                          padding='same',
                                          input_shape=xTrain.shape[1:]))
   autoencoder.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding='same'))

   # Flatten encoding for visualization
   autoencoder.add(tf.keras.layers.Flatten())
   autoencoder.add(tf.keras.layers.Reshape(target_shape=(112, 112, FILTER_DIM * 2)))

   # Decoder Layers
   autoencoder.add(tf.keras.layers.Conv2D(filters=FILTER_DIM * 2,
                                          kernel_size=(2, 2),
                                          activation='relu',
                                          padding='same'))
   autoencoder.add(tf.keras.layers.UpSampling2D(size=(2, 2)))

   autoencoder.add(tf.keras.layers.Conv2D(filters=1,
                                          kernel_size=(2, 2),
                                          activation='sigmoid',
                                          padding='same'))
   autoencoder.summary()

   autoencoder.compile(optimizer='adam',
                       loss='binary_crossentropy')
   autoencoder.fit(xTrain, yTrain,
                   epochs=10,
                   batch_size=4)
   return autoencoder


def preprocessData() -> None:
   """
   Preprocess data before training
   :return: None
   """

   print(f'preprocessing data...')
   downloadFiles()
   tf.enable_eager_execution()

   trainPaths, outlinesPaths, testPaths = getImagePaths()
   saveTrainData(trainImagePaths=trainPaths,
                 trainOutlinePaths=outlinesPaths)
   saveTestData(testImagePaths=testPaths)


def savePredictedImages(predicted: np.ndarray, testImages: np.ndarray) -> None:
   """
   Save predicted images to compare with test images

   :return: None
   """

   if not os.path.exists(os.path.join(os.getcwd(), 'result')):
      os.mkdir('./result')
   for i in range(len(predicted)):
      plt.rcParams["axes.grid"] = False
      plt.figure(figsize=(4, 8))
      fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
      ax[0].imshow(predicted[i, :, :, 0], cmap="gray")
      ax[1].imshow(testImages[i])
      plt.axis("off")
      plt.tight_layout()
      fileName = "./result/" + str(i) + ".png"
      fig.savefig(fname=fileName)


def startTraining() -> None:
   """
   Action

   :return: None
   """
   preprocessData()
   trainImages, trainOutlines = loadTrainData()
   testImages  = loadTestData()

   model       = defineModel(xTrain=trainImages, yTrain=trainOutlines)

   predicted   = model.predict(testImages)
   savePredictedImages(predicted, testImages)


if __name__ == '__main__':
   startTraining()

