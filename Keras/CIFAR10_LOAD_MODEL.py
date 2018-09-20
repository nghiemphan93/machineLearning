import numpy as np
import scipy.misc

# load model
from keras import Model
from keras.engine.saving import load_model

model: Model = load_model("CIFAR10_CONVNET.h5")



#load images
img_names = ['cat.jpg', 'dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32')
           for img_name in img_names]
imgs = np.array(imgs) / 255

results = model.predict(imgs)
index = np.argmax(results, axis=1)
print(index)