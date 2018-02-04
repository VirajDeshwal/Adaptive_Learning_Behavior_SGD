#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:37:47 2018

@author: virajdeshwal
"""

import numpy as np
import math

import keras
#import keras.backend as K
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint


batch_size = 32 
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
#breaking Training set into Training and Validation set
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# create and configure augmented image generator
datagen_train = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# create and configure augmented image generator
datagen_valid = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(x_train)
datagen_valid.fit(x_valid)


def build_model():
    net_input = Input(shape=x_train.shape[1:])
    
    net = Conv2D(32, (3, 3), padding='same', activation='relu')(net_input)
    net = Dropout(0.2)(net)
    
    net = Conv2D(32,(3,3),padding='same', activation='relu')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
 
    net = Conv2D(64,(3,3),padding='same',activation='relu')(net)
    net = Dropout(0.2)(net)
 
    net = Conv2D(64,(3,3),padding='same',activation='relu')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
 
    net = Conv2D(128,(3,3),padding='same',activation='relu')(net)
    net = Dropout(0.2)(net)
 
    net = Conv2D(128,(3,3),padding='same',activation='relu')(net)
    net = MaxPooling2D(pool_size=(2,2))(net)
    
    net = Flatten(name='flatten')(net) 
    net = Dense(1024, activation='relu')(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(256, activation='relu')(net)
    softmax_output = Dense(num_classes, activation='softmax')(net)

    model = Model(net_input, softmax_output)
    
    model.compile(optimizer=SGD(lr=0.01, momentum= 0, nesterov=True, decay=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return (model)


def decay_drop(epcoh):
    initial_lrate =0.01
    drop = 0.1
    epoch_drop = 5.0
    lrate = initial_lrate*math.pow(drop, math.floor((1+epochs)/epoch_drop))
    return lrate

lrate = LearningRateScheduler(decay_drop)
callback_list= [lrate]

model = build_model()

print(model.summary())
print("done")

model.fit(x_train, y_train, epochs=epochs, verbose=2, callbacks=callback_list,
          validation_data=(x_valid,y_valid), validation_split=0, shuffle=True,steps_per_epoch=2000,
          validation_steps= len(x_valid)//batch_size)
'''
checkpointer = ModelCheckpoint(filepath = 'model.weight.best.hdf5',verbose =1 , save_best_only=True)
hist = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2 ,callbacks = [checkpointer],
                    validation_data=datagen_valid.flow(x_valid, y_valid, batch_size=batch_size),
                    validation_steps=x_valid.shape[0] // batch_size)
print('Done.')
'''
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

