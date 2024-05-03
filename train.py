# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 9:24
# @Author  : yichongchen
# @purpose ：
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 17:50
# @Author  : yichongchen
# @purpose ：
# -*- coding: utf-8 -*-
# @Time    : 2023/11/26 15:19
# @Author  : yichongchen
# @purpose ：
'''
Good is 0.873070
'''

from __future__ import print_function
import os

import matplotlib
matplotlib.use('TKAgg')

from matplotlib import pyplot as plt
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling1D, UpSampling2D, MaxPooling1D, MaxPooling2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
import numpy as np
import tensorflow as tf
import os
import h5py
import scipy.io as sio
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
np.random.seed(7)


import math


# load training and test data and label

mat = h5py.File('./Traindata_demo.mat')

dataset=mat['Traindata_demo']
data= np.transpose(dataset[()])

x_train=data[0:1800,:,:,:]
x_test=data[1800:2000,:,:,:]
del mat
del dataset
del data
    
mat = h5py.File('./Trainlabel_demo.mat')

dataset=mat['Trainlabel_demo']
label= np.transpose(dataset[()])

y_train=label[0:1800,:,:]
y_test=label[1800:2000,:,:]

del mat
del dataset
del label   



print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# input image dimensions

img_rows, img_cols =64, 758
d1=[2,1,2,1,2,1,2] # 64-4
d2=[2,2,2,2,2,2,4] #758-3
u1=[2,2,2,1,2,1,2] #4-128

# the data, shuffled and split between train and test sets
y_train = y_train.reshape(y_train.shape[0], 128, 3, 1)
y_test = y_test.reshape(y_test.shape[0], 128, 3, 1)

input_shape = (img_rows, img_cols, 3)

# input image dimensions

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


batch_size = 8
epochs =100

main_input = Input(shape=input_shape, name='main_input')
print("main_input:",main_input.shape)
#第一个为128
x = Conv2D(128, kernel_size=(3, 3), padding='same')(main_input)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[0], d2[0]), padding='same')(x)  # 1
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("1:",x.shape)
#第二个为128
x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[1], d2[1]), padding='same')(x)  # 2
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("2:",x.shape)
#第三个256
x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[2], d2[2]), padding='same')(x)  # 3
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("3:",x.shape)
#第四个256
x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[3], d2[3]), padding='same')(x)  # 4
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("4:",x.shape)
#第五个512
x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[4], d2[4]), padding='same')(x)  # 5
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("5:",x.shape)
#第六个1024
x = Conv2D(1024, kernel_size=(3, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(d1[5], d2[5]), padding='same')(x)  # 6
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("6:",x.shape)

#第七个1024
x = Conv2D(1024, kernel_size=(1, 3), padding='same')(x)
#x = Dropout(0.5)(x)
x_encode = MaxPooling2D(pool_size=(d1[6], d2[6]), padding='same')(x)  # 7
x = LeakyReLU(alpha=0.2)(x_encode)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print(x_encode.shape)


#1024
x = Conv2D(1024, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[0], 1))(x)  # 1
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("1:",x.shape)
x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[1], 1))(x)  # 2
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("2:",x.shape)
x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[2], 1))(x)  # 3
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("3:",x.shape)
x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[3], 1))(x)  # 4
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("4:",x.shape)
x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[4], 1))(x)  # 5
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("5:",x.shape)
x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[5], 1))(x)  # 6
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
print("6:",x.shape)
x = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(u1[6], 1))(x)  # 7
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)

print("7:",x.shape)
x = Conv2D(16, kernel_size=(3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                       beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                       moving_variance_initializer='ones')(x)
main_output = Conv2D(1, kernel_size=(1, 3), padding='same')(x)
print(main_output.shape)


model = Model(inputs=[main_input], outputs=[main_output])
optimizer = keras.optimizers.adamax_v2.Adamax(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model.compile(loss='mse',  #cross entropy loss
              optimizer=optimizer,
              metrics=['accuracy'])

with tf.device("/gpu:0"):
    H = model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
#                             validation_split=0.2, shuffle=True)
                             validation_data=(x_test, y_test))


plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

    
loss1=H.history['loss']
np.savetxt("loss.txt",loss1)
loss2=H.history['val_loss']
np.savetxt("val_loss.txt",loss2)

# Open the save commond when you retrain the model
model.save('LC_FMNet.hdf5')

## validation prediction
y_pred = model.predict([x_test])

sio.savemat('./y_pred.mat',{'y_test':y_test,'y_pred':y_pred})





