#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, SpatialDropout2D
from keras import backend as K
from keras.models import Model
from PIL import Image
import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
from keras.optimizers import Adam, RMSprop, SGD
from keras_preprocessing.image import ImageDataGenerator

SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

data_path = '/home/yanzich/Desktop/Comp309Project/Train_data/'

def load_data():
   category=['cherry', 'strawberry', 'tomato']
   # use data augmentation to create more instance to train and validate the model
   # use split parameter to spilt the data into the training and validation
   data_gen = ImageDataGenerator(validation_split=0.2,
                                 height_shift_range=0.2,
                                 width_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
   # load the data from the training directory and set them to two flows training /validation
   training = data_gen.flow_from_directory(directory=data_path, target_size=(300,300),
                                           classes=category, batch_size=4, subset='training')
   test = data_gen.flow_from_directory(directory=data_path, target_size=(300,300),
                                       classes=category, batch_size=4, subset='validation')

   return training, test


def construct_model(flag):
   """
   Construct the CNN model.
   ***
       Please add your model implementation here, and don't forget compile the model
       E.g., model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])
       NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
   ***
   :return: model: the initial CNN model
   """
   model = None

   if flag == 'MLP':
       model = Sequential()
       model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu',padding='same'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Flatten())
       model.add(Dense(3))
       model.add(Activation('softmax'))
       # adam = Adam(lr=0.0001)
       # rms = RMSprop(lr=0.0001)
       sgd = SGD(lr = 0.001,momentum=0.5)
       model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
       # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])
       model.summary()

   # if flag == 'CNNwithDrop-Adam-orignal+Conv2D256':
   #     model = Sequential()
   #     model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #
   #     model.add(Conv2D(64, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Dropout(0.2))
   #
   #     model.add(Conv2D(256, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Dropout(0.2))
   #
   #     model.add(Flatten())
   #     model.add(Dense(128))
   #     model.add(Activation('relu'))
   #     model.add(Dropout(0.2))
   #     model.add(Dense(64))
   #     model.add(Activation('relu'))
   #     model.add(Dense(32))
   #     model.add(Activation('relu'))
   #     model.add(Dense(3))
   #
   #     model.add(Activation('softmax'))
   #
   #     adam = Adam(lr=0.0001)
   #     # rms = RMSprop(lr=0.0001)
   #     # sgd = SGD(lr=0.001, momentum=0.5)
   #     model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

   if flag == 'CNN1-5layers-newdropout':
       model = Sequential()

       model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(64, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(128, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(128, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(256, (3, 3)))
       model.add(Activation('relu'))
       model.add(SpatialDropout2D(0.2))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Flatten())
       model.add(Dense(256))
       model.add(Activation('relu'))
       model.add(Dropout(0.3))

       model.add(Dense(128))
       model.add(Activation('relu'))

       model.add(Dense(64))
       model.add(Activation('relu'))
       model.add(Dense(32))
       model.add(Activation('relu'))
       model.add(Dense(3))

       model.add(Activation('softmax'))

       adam = Adam(lr=0.001)
       # rms = RMSprop(lr=0.0001)
       # sgd = SGD(lr=0.001, momentum=0.5)
       model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
       # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])

   if flag == 'CNN2-5layers-SGD-originaldropout':
       model = Sequential()

       model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(64, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(128, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))

       model.add(Conv2D(128, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.2))

       model.add(Conv2D(256, (3, 3)))
       model.add(Activation('relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.2))

       model.add(Flatten())
       model.add(Dense(256))
       model.add(Activation('relu'))
       model.add(Dropout(0.2))
       model.add(Dense(128))
       model.add(Activation('relu'))
       model.add(Dense(64))
       model.add(Activation('relu'))
       model.add(Dense(32))
       model.add(Activation('relu'))
       model.add(Dense(3))
       model.add(Activation('softmax'))

       # adam = Adam(lr=0.001)
       # rms = RMSprop(lr=0.0001)
       sgd = SGD(lr=0.001, momentum=0.5)
       model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
       # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])

   # if flag == 'CNNwithDrop-SGD':
   #     model = Sequential()
   #     model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #
   #     model.add(Conv2D(64, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
       # model.add(MaxPooling2D(pool_size=(2, 2)))
       #
       # model.add(Conv2D(128, (3, 3)))
       # # model.add(BatchNormalization())
       # model.add(Activation('relu'))
       # model.add(MaxPooling2D(pool_size=(2, 2)))
       #
       # model.add(Conv2D(128, (3, 3)))
       # # model.add(BatchNormalization())
       # model.add(Activation('relu'))
       # model.add(MaxPooling2D(pool_size=(2, 2)))
       # model.add(Dropout(0.2))
       #
       # model.add(Conv2D(256, (3, 3)))
       # # model.add(BatchNormalization())
       # model.add(Activation('relu'))
       # model.add(MaxPooling2D(pool_size=(2, 2)))
       #
       # model.add(Flatten())
       # model.add(Dense(256))
       # model.add(Activation('relu'))
       # model.add(Dropout(0.2))
       # model.add(Dense(128))
       # model.add(Activation('relu'))
       # model.add(Dropout(0.2))
       # model.add(Dense(64))
       # model.add(Activation('relu'))
       # model.add(Dense(32))
       # model.add(Activation('relu'))
       # model.add(Dense(3))
       #
       # model.add(Activation('softmax'))
       #
       # # adam = Adam(lr=0.0001)
       # # rms = RMSprop(lr=0.0001)
       # sgd = SGD(lr=0.001, momentum=0.5)
       # model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
       # # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])
   # if flag == 'CNNwithoutDrop-withoutDense(256)-momentum0.4':
   #     model = Sequential()
   #     model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
   #
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(64, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(256, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #
   #     model.add(Flatten())
   #     model.add(Dense(128))
   #     model.add(Activation('relu'))
   #     model.add(Dense(64))
   #     model.add(Activation('relu'))
   #     model.add(Dense(32))
   #     model.add(Activation('relu'))
   #     model.add(Dense(3))
   #
   #     model.add(Activation('softmax'))
   #
   #     # adam = Adam(lr=0.0001)
   #     # rms = RMSprop(lr=0.0001)
   #     sgd = SGD(lr=0.001, momentum=0.4)
   #     model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])
   #     # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])
   # if flag == 'CNNwithoutDrop-momentum=0.6':
   #     model = Sequential()
   #     model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
   #
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(64, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Conv2D(128, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Dropout(0.2))
   #     model.add(Conv2D(256, (3, 3)))
   #     # model.add(BatchNormalization())
   #     model.add(Activation('relu'))
   #     model.add(MaxPooling2D(pool_size=(2, 2)))
   #     model.add(Dropout(0.2))
   #
   #     model.add(Flatten())
   #     model.add(Dense(128))
   #     model.add(Activation('relu'))
   #     model.add(Dense(64))
   #     model.add(Activation('relu'))
   #     model.add(Dense(32))
   #     model.add(Activation('relu'))
   #     model.add(Dense(3))
   #
   #     model.add(Activation('softmax'))
   #
   #     adam = Adam(lr=0.0001)
   #     # rms = RMSprop(lr=0.0001)
   #     # sgd = SGD(lr=0.001, momentum=0.6)
   #     model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
   #     # model.compile(rms, loss='categorical_crossentropy', metrics=['accuracy'])

   return model


def train_model(model,training,test):
   """
   Train the CNN model
   ***
       Please add your training implementation here, including pre-processing and training
   ***
   :param model: the initial CNN model
   :return:model:   the trained CNN model
   """
   # Add your code here
   import time
   from keras.callbacks import TensorBoard
   tb = TensorBoard(log_dir='./logs1{}'.format(time), write_graph=True, write_grads=True, write_images=True)
   model.fit_generator(training, validation_data = test ,
                       steps_per_epoch = 1500, epochs = 100,
                       verbose = 1,callbacks=[tb])
   return model

def save_model(model):
   """
   Save the keras model for later evaluation
   :param model: the trained CNN model
   :return:
   """
   # ***
   #   Please remove the comment to enable model save.
   #   However, it will overwrite the baseline model we provided.
   # ***

   model.save("model/model_CNN_Day3_2.h5")
   print("Model Saved Successfully.")

if __name__ == '__main__':

   training, test = load_data()
   model = construct_model('CNN2-5layers-SGD-originaldropout')
   model = train_model(model,training,test)
   save_model(model)

