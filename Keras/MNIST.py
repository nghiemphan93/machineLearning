from datetime import time
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import tensorflow as tf
from keras.callbacks import TensorBoard
from matplotlib import style
style.use('fivethirtyeight')


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))


train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = network.fit(train_images,
                      train_labels,
                      epochs=20,
                      batch_size=128,
                      validation_split=0.2)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('MNIST Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('MNIST Training and validation loss')
plt.legend()
plt.show()