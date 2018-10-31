import matplotlib.pyplot as plt
import keras.applications
from keras import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import style
style.use('fivethirtyeight')

print(len(vgg16Model.layers))

vgg16Model: Model = keras.applications.vgg16.VGG16(input_shape=(224, 224, 3))

model = Sequential()
for layer in vgg16Model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation="softmax"))
model.summary()

model.summary()
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/DogCat/cats_and_dogs_small/train"
validation_dir = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/DogCat/cats_and_dogs_small/validation"
test_dir = "C:/Users/Nghiem Phan/OneDrive - adesso Group/DataSet/DogCat/cats_and_dogs_small/test"

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              validation_data=validation_generator,
                              validation_steps=50,
                              epochs=10)


#model.save("dogCatFineTune.h5")


predictions = model.evaluate_generator(test_generator, steps=34, verbose=0)
print(predictions)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Dog Cat Fine Tune\nTraining and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Dog Cat Fine Tune\nTraining and validation loss')
plt.legend()
plt.show()