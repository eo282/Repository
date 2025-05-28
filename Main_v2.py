#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 14:24:52 2025

@author: eeerog
File to run UNet model training
takes input and groundtruth target for semantic segmentation task for classification into 3 classes
Save model
"""

#%%Modules
#standard modules
import random
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from numpy import save
import numpy.ma as ma
import glob
import math
from PIL import Image
import pandas as pd

#Statistics modules
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial import KDTree

#Deep learning modules
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import math
import keras
from keras  import backend as K

from keras.utils import plot_model as plt_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from keras.layers import *
from keras.models import *
from tensorflow.python.keras import backend as K
from tensorflow import keras

import tensorflow
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from PIL import Image
import pandas as pd
from os.path import exists
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import save
import numpy.ma as ma
import numpy as np
import gc

# WandB module
import wandb
wandb.init(entity='please give entity name', project='Project', group='Group name')
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import Callback
from wandb.integration.keras import WandbMetricsLogger

#function files
from train_model import *
from function_general_includingcoh import *

#%%Variables

#training data location
basic_location ='/home/eeerog/folder_to_end_all_folders/training_data/patched_64/'
groundtruth_input = basic_location + 'input/'
groundtruth_target = basic_location + 'gradient_target/'

model_saving_location = '/home/eeerog/folder_to_end_all_folders/models/'
model_name = 'model_name_'
class_nom = 3
cc_threshold = 0.1
number_to_test = 60
csv_directory = os.getcwd()
csv_file_namesn = 'csv'
percentage_validation = 0.2
batch_size = 8
epoch_list = [400, 425]

array_size = np.load(basic_location + 'input/file_1.npy').shape
input_list = os.listdir(groundtruth_input)
first = np.load(basic_location + 'input/file_1.npy')
first = np.expand_dims(first, axis = -1)
array_sizex, array_sizey, array_sizez = first.shape
array_size_indiv = array_sizex
shuffle_data = True

training_gradients = True

#%%Preparing Dataset
if training_gradients or training_gradients_and_noise or training_gradients_ft:
    train_samples_syn_noised, valid_samples_syn_noised,  nst, nsv = Lets_go_model_multi_coh(input_directory = groundtruth_input, target_directory = groundtruth_target,
                                                                              csv_directory=csv_directory, csv_file_names=csv_file_namesn,
                                                                              percentage_validation=percentage_validation, batch_size=batch_size, array_size=array_size,
                                                                              shuffle_data=shuffle_data, class_nom=class_nom)
    
    train_samples_syn = train_samples_syn_noised 
    valid_samples_syn = valid_samples_syn_noised
    
    class_weightsh1, class_weightsc = listing_function(basic_location, class_nom, array_size_indiv)

if training_gradients:
    train_datagen = samples_gen_part_two_output_coh(samples=train_samples_syn, batch_size=batch_size, shuffle_data=shuffle_data, array_size=array_size, class_nom=class_nom)
    valid_datagen = samples_gen_part_two_output_coh(samples=valid_samples_syn, batch_size=batch_size, shuffle_data=shuffle_data, array_size=array_size, class_nom=class_nom)

    model = CustomModel()
    model = model.assemble_full_model()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6),  # Fix this line
        loss={
            'finalx': keras.losses.SparseCategoricalCrossentropy(),
        }
    )
    
    nst_steps = (nst - 1)/batch_size
    nsv__steps = (nsv - 1)/batch_size
    history = model.fit(train_datagen, epochs=epochs,
                        steps_per_epoch=int(nst_steps), callbacks=[reduce_lr_on_plateau, early_stopping, WandbMetricsLogger()], validation_data=valid_datagen, validation_steps=int(nsv__steps))
    history.history.keys()
    model.save(model_saving_location + model_name + str(epochs) + '.keras')
