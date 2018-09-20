import shutil
import cv2, os
import numpy as np

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
    dogs = np.zeros((DOG_NUMB, 150, 150, 3))
    cats = np.zeros((CAT_NUMB, 150, 150, 3))

    sampleIndex = 0
    for filename in os.listdir(folder):
        if "dog" in filename:
            dogs[sampleIndex] = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float32)])
        if "cat" in filename:
            dogs[sampleIndex] = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float32)])
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

#(train_images, train_labels), (test_images, test_labels)
#np.savez("cats_dogs_train.npz", catsTrain=cats, dogsTrain=dogs)
#cats, dogs = loadDogsCats(original_dataset_dir)

'''
data = np.load("cats_dogs_train.npz")
cats = data["catsTrain"]
dogs = data["dogsTrain"]
'''
cats = np.zeros((12500, 150, 150, 3))
dogs = np.zeros((12500, 150, 150, 3))

testLabel = createTestLabel(cats, dogs)
print(testLabel.shape)
print(testLabel)

np.savez_compressed("testLabel.npz", testLabel=testLabel)

data = np.load("cats_dogs_train.npz")
cats = data["catsTrain"]
dogs = data["dogsTrain"]
trainData = np.vstack(cats, dogs)

np.savez_compressed("trainLabel")

np.savez_compressed("train.npz", trainData=trainData, trainLabel=testLabel)

