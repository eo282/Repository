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


#%%Loss functions

def wsse(class_weights, num_classes):
    def metric(y_true, y_pred):
        #make sure the groundtruth arrays are the correct size
        y_true = tf.squeeze(y_true, axis=-1) 
        y_true = tensorflow.cast(y_true, dtype=tensorflow.int32)
        reduction = tensorflow.keras.losses.Reduction.NONE
        #perform the normal sparse categorical crossentropy loss calculation
        unreduced_scce = tensorflow.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name='weighted_scce', reduction=reduction)
        loss = unreduced_scce(y_true, y_pred)
        #add the weighting to the loss calculation
        weight_mask = tensorflow.gather(class_weights, y_true)
        weight_mask = tensorflow.cast(weight_mask, tensorflow.float32)
        loss = tensorflow.math.multiply(loss, weight_mask)
        return tensorflow.reduce_mean(loss)
    return metric

def dice_coefficient(num_classes, smooth=1):
    '''
    Parameters:
        Number of class: class number for classification task
        smooth: Smoothing factor, higher value is more smooth
        
    '''
    def metric(y_true, y_pred):
        dice_scores = []
        #make sure the groundtruth is the correct size
        y_true = tf.squeeze(y_true, axis=-1)  # Ensure shape matches
        for class_id in range(0, num_classes):
            #for each class, calculate the dice coefficient loss
            y_true_class = tensorflow.cast(tensorflow.equal(
                y_true, class_id), dtype=tensorflow.float32)
            y_pred_class = y_pred[..., class_id]
            intersection = tensorflow.reduce_sum(y_true_class * y_pred_class)
            union = tensorflow.reduce_sum(tensorflow.cast(y_true_class, dtype=tensorflow.float32)) + tensorflow.reduce_sum(
                tensorflow.cast(y_pred_class, dtype=tensorflow.float32))  # Cast y_pred_class
            dice = 1 - ((2. * intersection + 1) / (union + 1))
            
            #append calculated loss per class to the total dice score                
            dice_scores.append(dice)
        #return the average dice loss across the entire interferogram
        return tensorflow.reduce_mean(dice_scores)

    return metric

def custom_loss(class_weights, num_classes):
    
    '''
    Custom loss to combine together multiple loss functions for optimal loss training
    Input:
        Class weights: List containing class weights per class to be used for the weighted categorical crossentropy
        number of classes: integer value of the number of classes to be used for training
       
    '''

    def loss(y_true, y_pred):
        
        dice_coef_loss = dice_coefficient(num_classes, class_weights)(y_true, y_pred)
        wsse_loss = wsse(class_weights, num_classes)(y_true, y_pred)
        total_loss =  wsse_loss + dice_coef_loss
        
        y_true = tf.one_hot(tensorflow.cast(y_true, dtype=tensorflow.int32), depth=4) 
        tvloss = tf.keras.losses.tversky(y_true, y_pred, alpha=0.7, beta=0.3)

        return total_loss + tvloss
    return loss



#%%Model
class CustomModel():
    '''
    Custom Model for training
    Paramters:
        num_classes: integer value for the number of classification values 
    '''
    def attention_gate(self, g, s, num_filters):
        Wg = Conv2D(num_filters, 1, padding="same")(g)
        Wg = BatchNormalization()(Wg)
    
        Ws = Conv2D(num_filters, 1, padding="same")(s)
        Ws = BatchNormalization()(Ws)
    
        out = Activation("relu")(Wg + Ws)
        out = Conv2D(num_filters, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
    
        return out * s

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
        # UP1_B = Concatenate()([UP1, BN7])
        UP1_B = self.attention_gate(UP1, BN7, 512)
        Conv4_1 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv4_1')(UP1_B)
        BN8 = BatchNormalization()(Conv4_1)
        Conv4_2 = Conv2D(512, kernel_size, activation='relu', padding='same', name='conv4_2')(BN8)
        BN9 = BatchNormalization()(Conv4_2)
        
        UP2 = UpSampling2D()(BN9)
        # UP2_B = Concatenate()([UP2, BN5])
        UP2_B = self.attention_gate(UP2, BN5, 256)
        Conv5_1 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv5_1')(UP2_B)
        BN10 = BatchNormalization()(Conv5_1)
        Conv5_2 = Conv2D(256, kernel_size, activation='relu', padding='same', name='conv5_2')(BN10)
        BN11 = BatchNormalization()(Conv5_2)
        
        UP3 = UpSampling2D()(BN11)
        # UP3_B = Concatenate()([UP3, BN3])
        UP3_B = self.attention_gate(UP3, BN3, 128)
        Conv6_1 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv6_1')(UP3_B)
        BN12 = BatchNormalization()(Conv6_1)
        Conv6_2 = Conv2D(128, kernel_size, activation='relu', padding='same', name='conv6_2')(BN12)
        BN13 = BatchNormalization(name='bn')(Conv6_2)
        
        convadd1x = Conv2D(128, kernel_size, activation='relu', padding='same', name='xdirection_extraconv')(BN13)
        convadd1y = Conv2D(128, kernel_size, activation='relu', padding='same', name='ydirection_extraconv')(BN13)
        convadd1x2 = Conv2D(64, kernel_size, activation='relu', padding='same', name='xdirection_extraconv2')(convadd1x)
        convadd1y2 = Conv2D(64, kernel_size, activation='relu', padding='same', name='ydirection_extraconv2')(convadd1y)

        return convadd1x2, convadd1y2

    def assemble_full_model(self, num_classes=4):
        input_size = (None, None, 1)
        inputs = tensorflow.keras.Input(shape=input_size)

        conv17o, conv17br1 = self.make_default_hidden_layers(
            inputs, num_classes=num_classes, kernel_size=3, dropout_value=0.2)
        
        conv17 = Conv2D(num_classes, 1, padding='same',
                        activation='softmax', name='finalx')(conv17o)
        
        conv17b = Conv2D(num_classes, 1, padding='same',
                          activation='softmax', name='finaly')(conv17br1)


        model = Model(inputs=inputs, outputs= (conv17, conv17b))

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

