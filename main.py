from keras.datasets import boston_housing
from sklearn.datasets import load_boston
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

'''
# Load data
(trainData, trainTarget), (testData, testTarget) = boston_housing.load_data()

# Normalize Data
mean = trainData.mean(axis=0)
std = trainData.std(axis=0)
trainData = (trainData - mean) / std
testData = (testData - mean) / std

# Build model
model = Sequential()
model.add(Dense(units=64, activation="relu", input_shape=(13,)))
model.add(Dense(units=80, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",
              loss="mae",
              metrics=["mae"])
history = model.fit(x=trainData, y=trainTarget, epochs=60, validation_split=0.2, batch_size=32)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(model.evaluate(x=testData, y=testTarget))

# boston= load_boston()
# df = pd.DataFrame(data=boston.get("data"), columns=boston.get("feature_names"))
# price = pd.DataFrame(data=boston.get("target"), columns=["price"])
# df: pd.DataFrame = pd.concat(objs=[df, price], axis=1)
# print(df.describe())
'''



(trainData, trainTarget), (testData, testTarget) = mnist.load_data()

trainData = trainData.reshape((len(trainData), 28, 28, 1))
trainData = trainData / 255.0

testData = testData.reshape((len(testData), 28, 28, 1))
testData = testData / 255.0


trainTarget = to_categorical(trainTarget)
testTarget = to_categorical(testTarget)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
history = model.fit(trainData, trainTarget, epochs=10, batch_size=1024, validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()












