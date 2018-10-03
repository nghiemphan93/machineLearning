import numpy as np
import cv2, os, shutil
from keras.models import load_model
from keras import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt



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



model: Model = load_model("zahlErkennung3.h5")

testFolder = "./testNumber/"
images = loadImages(testFolder)
print(images.shape)
results = model.predict(images)
print(np.argmax(results, axis=1) + 10)

