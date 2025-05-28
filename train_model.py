#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:44:53 2025

@author: eeerog

Model location files to be called to allow model training

Model file
"""

#%%Module load

import tensorflow
import numpy as np
from tensorflow import keras as keras
from keras  import backend as K
from keras.utils import plot_model as plt_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.layers import *
from keras.models import *
from tensorflow.python.keras import backend as K
import tensorflow as tf



#%%Model
class UNet():
    '''
    Custom Model for training - based upon Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation paper
    Paramters:
        num_classes: integer value for the number of classification values 
    '''


    def make_default_hidden_layers(self, inputs, num_classes, kernel_size=3, dropout_value=0.2):
        # Initial Convolutional Block
        Conv1_1 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv1_1')(inputs)
        BN1 = BatchNormalization()(Conv1_1)
        Conv1_2 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv1_2')(BN1)
        BN2 = BatchNormalization()(Conv1_2)
        Conv1_3 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv1_3')(BN2)
        BN3 = BatchNormalization()(Conv1_3)
        MP1 = MaxPooling2D(pool_size=(2, 2))(BN3)
        
        # Downsampling Blocks
        Conv2_1 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv2_1')(MP1)
        BN4 = BatchNormalization()(Conv2_1)
        Conv2_2 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv2_2')(BN4)
        BN5 = BatchNormalization()(Conv2_2)
        MP2 = MaxPooling2D(pool_size=(2, 2))(BN5)
        
        Conv3_1 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv3_1')(MP2)
        BN6 = BatchNormalization()(Conv3_1)
        Conv3_2 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv3_2')(BN6)
        BN7 = BatchNormalization()(Conv3_2)
        MP3 = MaxPooling2D(pool_size=(2, 2))(BN7)
        
        Conv3_1d = Conv2D(1024, kernel_size, activation='relu', padding='same', name='conv3_1d')(MP3)
        BN6_d = BatchNormalization()(Conv3_1d)
        Conv3_1d2 = Conv2D(1024, kernel_size, activation='relu', padding='same', name='conv3_1d2')(BN6_d)
        BN6_d2 = BatchNormalization()(Conv3_1d2)
        Conv3_1d2 = Conv2D(1024, kernel_size, activation='relu', padding='same', name='conv3_1d222')(BN6_d2)
        BN6_d2 = BatchNormalization()(Conv3_1d2)
        Conv3_1d2 = Conv2D(1024, kernel_size, activation='relu', padding='same', name='conv3_1d22')(BN6_d2)
        BN6_d2 = BatchNormalization()(Conv3_1d2)
        dropout = Dropout(rate = 0.1)(Conv3_1d2)
        
        #Upsampling blocks
        UP1 = UpSampling2D()(dropout)
        UP1_B = Concatenate()([UP1, BN7])

        Conv4_1 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv4_1')(UP1_B)
        BN8 = BatchNormalization()(Conv4_1)
        Conv4_2 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv4_2')(BN8)
        BN9 = BatchNormalization()(Conv4_2)
        
        UP2 = UpSampling2D()(BN9)
        UP2_B = Concatenate()([UP2, BN5])

        Conv5_1 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv5_1')(UP2_B)
        BN10 = BatchNormalization()(Conv5_1)
        Conv5_2 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv5_2')(BN10)
        BN11 = BatchNormalization()(Conv5_2)
        
        UP3 = UpSampling2D()(BN11)
        UP3_B = Concatenate()([UP3, BN3])
        
        Conv6_1 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv6_1')(UP3_B)
        BN12 = BatchNormalization()(Conv6_1)
        Conv6_2 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv6_2')(BN12)
        BN13 = BatchNormalization(name='bn')(Conv6_2)
        
        convadd1x = Conv2D(128, kernel_size, activation='relu', padding='same', name='xdirection_extraconv')(BN13)
        convadd1x2 = Conv2D(64, kernel_size, activation='relu', padding='same', name='xdirection_extraconv2')(convadd1x)

        return convadd1x2

    def assemble_full_model(self, num_classes=3):
        input_size = (None, None, 1)
        inputs = tensorflow.keras.Input(shape=input_size)

        conv17o = self.make_default_hidden_layers(
            inputs, num_classes=num_classes, kernel_size=3, dropout_value=0.2)
        
        conv17 = Conv2D(num_classes, 1, padding='same',
                        activation='softmax', name='finalx')(conv17o)
        

        model = Model(inputs=inputs, outputs= conv17)

        return model

#%%Callbacks

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increase patience
    restore_best_weights=True
)

reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,  # Reduce patience so LR decreases first
    min_lr=1e-9
)

