#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:29:16 2018

@author: virajdeshwal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:15:44 2017

@author: virajdeshwal
"""

from keras.datasets import cifar10
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import math
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize= (20,5))
for i in range(36):
    ax =fig.add_subplot(3, 12, i+1, xticks=[], yticks= [])
    ax.imshow(np.squeeze(x_train[i]))
    

#one-hot encode the labels 
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# break the training set in training and validation 
(x_train , x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#print the shape of the training set
print('X_training set is here hurrraaaaayyyyy!!!  : ', x_train.shape)
print('X_training set no. are here hurrraaaaayyyyy!!!  : ', x_train.shape[0])
print('y_training set no. are here hurrraaaaayyyyy!!!  : ', x_test.shape[0])
print('X_validation set no. are here hurrraaaaayyyyy!!!  : ', x_valid.shape[0])

#define the model architecure
epochs=20
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size =2,strides=1, padding = 'same', activation= 'relu' , input_shape =(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters =32, kernel_size=2,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2,strides=1,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))    
model.add(Dropout(0.3))


model.add(Flatten())

model.add(Dense(500, activation = 'relu'))

model.add(Dropout(0.4))

#predicting for 10 classess. So the final layer contain 10 layers
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(loss= 'categorical_crossentropy', optimizer = SGD(lr=0.0, momentum=0, decay=0.0, nesterov=True), metrics=['accuracy'])

def decay_drop(epoch):
    initial_lrate =0.1
    drop = 0.1
    epoch_drop = 5
    lrate = initial_lrate*math.pow(drop, math.floor((1+epochs)/epoch_drop))
    return (lrate)
lrate = LearningRateScheduler(decay_drop)
callback_list= [lrate]


hist = model.fit(x_train, y_train, batch_size =32, epochs =epochs, validation_data=(x_valid, y_valid),
                 callbacks=callback_list,
                  verbose =2, shuffle =True)


score = model.evaluate(x_test, y_test, verbose=2)
print('\n test accuracy is {}  lol dont laugh'.format(score[1]))

#wooho lets predict the worst prediction of my data 

y_hat = model.predict(x_test)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
