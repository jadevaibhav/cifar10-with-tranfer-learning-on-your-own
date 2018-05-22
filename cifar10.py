from keras.datasets import cifar10
import numpy as np
(x_train, y_train), (x_val, y_val) = cifar10.load_data()

Y_train = np.zeros((y_train.shape[0],10))
for i in range(y_train.shape[0]):
    Y_train[i][y_train[i]] = 1
Y_val = np.zeros((y_val.shape[0],10))
for i in range(y_val.shape[0]):
    Y_val[i][y_val[i]] = 1

import tensorflow as tf
import keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32,kernel_size=(7,7),padding='same',strides=(2,2),activation='relu',input_shape=(32,32,3)))
model.add(layers.Conv2D(32,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(32,kernel_size=(7,7),padding='same',strides=(2,2),activation='relu'))
model.add(layers.Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(32,kernel_size=(7,7),strides=(1,1),padding='same',activation='relu'))
model.add(layers.Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss=keras.losses.squared_hinge,
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

model.fit(x_train, (Y_train),
          batch_size= 32,
          epochs=10,
          verbose=1,
          validation_data=(x_val, (Y_val)))

model.save('cifar10_new_model_10.h5')
