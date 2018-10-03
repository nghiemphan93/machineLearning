import os
import numpy as np
import cv2



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
        if i < 12500: testLabel[i] = 0
        else: testLabel[i] = 1
    return testLabel