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

original_dataset_dir = 'C:/Users/phan/Downloads/DataSet/DogCat/train'
base_dir = 'C:/Users/phan/Downloads/DataSet/DogCat/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
testSamples = os.path.join(test_dir, "test")


def loadImagesFromFolder(folder: str):
    images = np.zeros((len(os.listdir(folder)), 150, 150, 3))
    sampleIndex = 0
    for filename in os.listdir(folder):
        img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float32)])
        images[sampleIndex] = img
    return images

def countDogsCats(folder: str):
    CAT_NUMB = 0
    DOG_NUMB = 0
    for filename in os.listdir(folder):
        if "dog" in filename:
            DOG_NUMB += 1
        if "cat" in filename:
            CAT_NUMB +=1
    return CAT_NUMB, DOG_NUMB

def loadDogsCats(folder: str):
    CAT_NUMB, DOG_NUMB = countDogsCats(original_dataset_dir)
    dogs = np.zeros((DOG_NUMB, 150, 150, 3), dtype="float16")
    cats = np.zeros((CAT_NUMB, 150, 150, 3), dtype="float16")

    dogIndex = 0
    catIndex = 0
    for filename in os.listdir(folder):
        if "dog" in filename:
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float16)])
            img = img / 255
            dogs[dogIndex] = img
            dogIndex += 1
            print(img)
        if "cat" in filename:
            img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float16)])
            img = img / 255
            cats[catIndex] = img
            catIndex += 1
            print(img)
        print(dogIndex, " ", catIndex)
    return cats, dogs

def createTestLabel(catsTrain, dogsTrain):
    temp = len(catsTrain) + len(dogsTrain)
    testLabel = np.zeros((temp, 2))
    print(type(testLabel))
    print(testLabel.shape)
    for i in range(len(catsTrain)):
        testLabel[i] = np.array([0, 1])

    for j in range(len(catsTrain), len(catsTrain) + len(dogsTrain)):
            testLabel[j] = np.array([1, 0])
    return testLabel

def createTestLabel2(trainData):
    temp = len(trainData)
    testLabel = np.zeros((temp, 1))
    print(type(testLabel))
    print(testLabel.shape)
    for i in range(len(trainData)):
        if i < 12500: testLabel[i] = 0  # cat
        else: testLabel[i] = 1          # dog
    return testLabel

#(train_images, train_labels), (test_images, test_labels)
#np.savez("cats_dogs_train.npz", catsTrain=cats, dogsTrain=dogs)
#cats, dogs = loadDogsCats(original_dataset_dir)

'''
data = np.load("cats_dogs_train.npz")
cats = data["catsTrain"]
dogs = data["dogsTrain"]
'''

'''
cats = np.zeros((12500, 150, 150, 3))
dogs = np.zeros((12500, 150, 150, 3))

testLabel = createTestLabel(cats, dogs)
np.savez_compressed("testLabel.npz", testLabel=testLabel)
'''
'''
data = np.load("cats_dogs_train.npz")
cats = data["catsTrain"]
dogs = data["dogsTrain"]
trainData = np.vstack(cats, dogs)
'''

'''
cats, dogs = loadDogsCats(original_dataset_dir)
print("Cats: ", cats)
print("Dogs: ", dogs)
trainData = np.vstack([cats, dogs])
print(trainData)
trainLabel = createTestLabel2(trainData)
np.savez_compressed("train.npz", trainData=trainData, trainLabel=trainLabel)
'''



'''
data = np.load("train.npz")
trainData, trainLabel = data["trainData"], data["trainLabel"]
trainLabel = createTestLabel2(trainData)
print("train label shape: ", trainLabel.shape)
print(trainLabel)

np.savez_compressed("train.npz", trainData=trainData, trainLabel=trainLabel)
'''


data = np.load("train4000Samples.npz")
trainData = data["trainData"]
trainLabel = data["trainLabel"]


'''
data = np.load("train.npz")
trainData = data["trainData"]
trainLabel = data["trainLabel"]

idx = np.random.permutation(len(trainData))
trainData, trainLabel = trainData[idx], trainLabel[idx]
print(trainLabel)
np.savez_compressed("train4000Samples.npz", trainData=trainData[:4000], trainLabel=trainLabel[:4000])
'''


model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.01),
              metrics=['acc'])
model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
history = model.fit(trainData, trainLabel,
                    epochs=30,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[tensorboard])


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
model.save("dogcat.h5")
