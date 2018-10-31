import numpy as np
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

def build_model():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    x_val = train_data[:50]
    partial_x_train = train_data[50:]

    y_val = train_targets[:50]
    partial_y_train = train_targets[50:]


    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu',
                           input_shape=(13, )))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])


    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=12,
                        validation_data=(x_val, y_val))
    results = model.evaluate(test_data, test_targets)
    print(model.predict(test_data))
    print(results)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('House Price\n Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def loadData():
   (trainData, trainTarget), (testData, testTarget) = boston_housing.load_data()
   print(trainData.shape[1])
   print(trainTarget)

build_model()




