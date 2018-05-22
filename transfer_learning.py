import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras import backend as K
model = keras.models.load_model('mnist_20.h5')
model.summary()

#taking cifar10 dataset
from keras.datasets import cifar10
import numpy as np
(x_train, y_train), (x_val, y_val) = cifar10.load_data()

Y_train = np.zeros((y_train.shape[0],10))
for i in range(y_train.shape[0]):
    Y_train[i][y_train[i]] = 1
Y_val = np.zeros((y_val.shape[0],10))
for i in range(y_val.shape[0]):
    Y_val[i][y_val[i]] = 1

#function defining transfer_learning
def transfer_learning(num_classes,input_shape,model,model_input_shape):
    X_input = layers.Input(input_shape)
    X = layers.Conv2D(model_input_shape[2],kernel_size=(input_shape[1]-model_input_shape[1],input_shape[1]-model_input_shape[1]),strides=(1,1))(X_input)
    model.pop()
    X = model(inputs=X)
    X = layers.Dense(num_classes,activation='softmax')(X)

    model = keras.Model(inputs=X_input,outputs=X,name='transfer_learning')

    return model
#creating and training the model
ci10 = transfer_learning(10,(32,32,3),model,(28,28,1))
ci10.summary()

ci10.compile(loss=keras.losses.squared_hinge,
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

ci10.fit(x_train, (Y_train),
          batch_size= 32,
          epochs=10,
          verbose=1,
          validation_data=(x_val, (Y_val)))

ci10.save('cifar10_tranfer_10.h5')
