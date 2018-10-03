from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

'''
imgPath     = "./testNumber/10_200.jpg"
imgPath2    = "./testNumber/14_266.jpg"
img         = image.load_img(imgPath, target_size=(56, 56))
img2        = image.load_img(imgPath2, target_size=(56, 56))
img_tensor  = image.img_to_array(img)
img_tensor2  = image.img_to_array(img2)
print("Shape 1: ", img_tensor.shape)

img_tensor  = np.expand_dims(img_tensor, axis=0)
img_tensor2  = np.expand_dims(img_tensor2, axis=0)
img_tensor  = img_tensor/ 255.0
img_tensor2  = img_tensor2/ 255.0

print("Shape 2: ", img_tensor.shape)

imgTensors = np.vstack([img_tensor, img_tensor2])

print("Shape Tensors: ", imgTensors.shape)

plt.imshow(imgTensors[1])
plt.show()
'''

testFolder = "./testNumber"
images = np.zeros((len(os.listdir(testFolder)), 56, 56, 3))
i = 0
for filename in os.listdir(testFolder):
    img = image.load_img(os.path.join(testFolder, filename), target_size=(56, 56))
    imgArray = image.img_to_array(img)
    imgArray = imgArray / 255.0
    images[i] = imgArray
    i += 1

print(images)
print(images.shape)

plt.imshow(images[0])
plt.show()

plt.imshow(images[1])
plt.show()

plt.imshow(images[2])
plt.show()