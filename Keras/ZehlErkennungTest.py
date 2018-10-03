import numpy as np
import cv2, os, shutil
from keras.models import load_model
from keras import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt


'''
def loadImagesFromFolder(folder: str):
    images = np.zeros((len(os.listdir(folder)), 150, 150, 3))
    sampleIndex = 0
    for filename in os.listdir(folder):
        img = np.array([cv2.resize(cv2.imread(os.path.join(folder, filename)), (150, 150)).astype(np.float16)])
        images[sampleIndex] = img
        sampleIndex += sampleIndex
    return images


testImages = loadImagesFromFolder("./testDogCat")
print(testImages)
print(testImages.shape)
'''

def loadImages(folder: str):
    images = np.zeros((len(os.listdir(folder)), 56, 56, 3))
    i = 0
    for filename in os.listdir(folder):
        img = image.load_img(os.path.join(folder, filename), target_size=(56, 56))
        imgArray = image.img_to_array(img)
        imgArray = imgArray / 255.0
        images[i] = imgArray
        i += 1
    return images




model: Model = load_model("steffanNumbers.h5")
'''
model.compile(loss='binary_crossentropy',
              optimizer="rmsprop",
              metrics=['acc'])
'''

'''
# predicting images
img = image.load_img('./testNumber/10_200.jpg', target_size=(56, 56))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

results = model.predict(images)
print(results)
'''

testFolder = "./testNumber/"
images = loadImages(testFolder)
print(images)
results = model.predict(images)
print(np.argmax(results, axis=1))

print(model.layers)

#print(decode_predictions(results, top=1))




'''
classes = model.predict_classes(images, batch_size=10)
results = model.predict(images)
print(results)
print(classes)
'''