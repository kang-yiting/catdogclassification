# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:17:27 2019

@author: user
"""

import os #系統路徑相關的模組
import glob
import numpy as np #數列以及array相關的功能
from keras.preprocessing.image import  img_to_array, load_img #把圖檔轉換為array、讀取圖檔
from PIL import Image

images_all=[]
labels_all=[]
#Read cat npy file
images_cat=np.load('D:\kagglecatsanddogs_3367a\Cat_Dog_Dataset\Cat_images.npy')
images_all=images_cat
labels_cat=np.load('D:\kagglecatsanddogs_3367a\Cat_Dog_Dataset\Cat_labels_hot.npy')
labels_all=labels_cat
#Read dog npy file
images_dog=np.load('D:\kagglecatsanddogs_3367a\Cat_Dog_Dataset\Dog_images.npy')
images_all=np.append(images_all,images_dog,axis=0)
labels_dog=np.load('D:\kagglecatsanddogs_3367a\Cat_Dog_Dataset\Dog_labels_hot.npy')
labels_all=np.append(labels_all,labels_dog,axis=0)

#%% VGG 16 only for classification
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
# Generate model
model = Sequential()
# input: 190x190 images with 3 channels -> (190, 190, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,3),padding='same',name='block1_conv2_1'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
model.add(Dropout(0.25))
#
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_1'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_2'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block3_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block4_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block5_MaxPooling'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='final_output_1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu',name='final_output_2'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid',name='class_output'))
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
EStop = EarlyStopping(monitor='val_acc', min_delta=0, 
                      patience=10, verbose=1, mode='auto')

model.fit(train_data, train_label, batch_size=64, epochs=100,shuffle=True, validation_split=0.2,callbacks=[EStop])