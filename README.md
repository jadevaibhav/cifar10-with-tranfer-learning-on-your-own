# cifar10 with Transfer Learning(on your own)
Fun continued with cifar10 dataset using cnn in Keras, also use transfer learning from your own model

## Getting Started 
Before starting with this tutorial, you should already have tensorflow and Keras installed(with Python 3). Like mnist_trial, this repository contains 2 files cifar10.py and transfer_learning.py. 

## Data Specifaction
I have used cifar 10 pre-processed dataset provided by Keras. The labels(i.e., y_train and y_val) are numbered from 0 to 9(for 10 classes). I have created one_hot lables for them.

## cifar10.py
File contains the above mentioned preprocessing, model arcitecture defination and training. Trained model would be saved in cifar10_new_model_10.h5. I have used batch size of 32 and trained the model for 10 eopchs.

## tranfer_learning.py
Rather than using pre-trained models provided by Keras(or other sources), I have used custom model previously trained on mnist dataset and used it here. As the model is much more simpler(shallow) than these well-known models, accuracy reached is much lesser. But I wanted to show how to use your own pre-trained model for transfer learning rather than the provided limited options. You first load the model
```
model = keras.models.load_model('mnist_20.h5')
```
and then use the function transfer_learning.
```
def transfer_learning(num_classes,input_shape,model,model_input_shape):
```
```
Arguments:
num_classes : # of classes for classification
input_shape : input shape of image sample to be classified(must be more than model_input_shape)
model : model which would be used for tranfer learning
model_input_shape : input shape used for the model
```
I have used Keras functional API which allows non-sequential flow of graphs. To resize the image from input shape to model input shape, rather than cropping the image, I have added a convolutional layer, which would rather create embedding into smaller dimension without loss of data as to cropping.
```
layers.Conv2D(model_input_shape[2],kernel_size=(input_shape[1]-model_input_shape[1],input_shape[1]-model_input_shape[1]),strides=(1,1))(X_input)
```
