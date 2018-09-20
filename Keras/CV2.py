import cv2, numpy as np
from keras import Model
from keras.engine.saving import load_model

HEIGHT      = 32
WIDTH       = 32
CHANNEL     = 3
NUMB_LABEL  = 10
VERBOSE     = 1
BATCH_SIZE  = 128
NUMB_EPOCHS = 10

img = np.array([cv2.resize(cv2.imread("dog.jpg"), (HEIGHT, WIDTH)).astype(np.float32)])
img = img / 255.0

print(img)
print(img.shape)

model: Model = load_model("CIFAR10_CONVNET.h5")

result = model.predict(img)

print(np.argmax(result))