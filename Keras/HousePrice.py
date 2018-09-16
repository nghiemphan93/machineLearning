import numpy as np
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

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

model = build_model()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=500,
                    batch_size=12,
                    validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



results = model.evaluate(test_data, test_targets)
print(model.predict(test_data))
print(results)