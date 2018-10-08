from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import keras.applications
from keras import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os

'''
test_dir = "C:/Users/phan/Downloads/DataSet/DogCat/cats_and_dogs_small/test"
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical')

images, label = next(test_generator)


for i in range(len(images)):
    image = images[i]
    image = image / 255
    plt.imshow(image)
    plt.show()
    print(label[i])
'''

generator = ImageDataGenerator(rotation_range=180,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               channel_shift_range=10,
                               horizontal_flip=True)
imagePath = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/DogCat/cats_and_dogs_small/lon/cat/cat.42.jpg"

# Obtain imag
img = image.load_img(imagePath, target_size=(244, 244))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)


augIter = generator.flow(img,
                         save_to_dir="C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/DogCat/cats_and_dogs_small/lon/augment", save_format="jpg",
                         )



i = 0
while(True):
   img = next(augIter)[0]
   img = img / 255
   i = i + 1
   if i == 500:
      break
