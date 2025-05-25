#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:48:40 2023

@author: eeerog
File for plotting gradients 
Will show where your model is predicting correctly and where it is predicting incorrectly
"""

#%%Load in modules
import math
import seaborn as sns
import os
import numpy as np
import ntpath

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation


import tensorflow
tensorflow.config.set_visible_devices([], 'GPU')
import wandb
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
import gc

#%%Variables
model_4 = '/home/eeerog/folder_to_end_all_folders/chapter_1/models/Keep_4Class_1e_5_64batchsize_unfilteredtarget_noisdatatrained50noise_Set_to_0.model'
Model_location = [model_4]
output_class_nom = [4]
model_names = ['4Class_1e_5_64batchsize_unfilteredtarget_noisdatatrained25noise_Set_to_0 output_NOISEDVERSION']

file_location_base = '/home/eeerog/Downloads/OneDrive_1_10-04-2025/testing_dataset/ambiguity_gradient/ready/'
output_file_location = '/home/eeerog/folder_to_end_all_folders/chapter_1/analysing_results/'

list_dir = os.listdir(file_location_base + 'wrapped/')
x_dict = {}
y_dict = {}

#%%Processing Functions
def expanded_array_func_x(array):
    
    '''
    Expanding array in the horizontal direction function:
        Parameters:
            array: 2d numpy array
    '''
    original_array = array
    l, w = array.shape
    
    # Create a new  array filled with zeros bigger along axis = 1
    new_array = np.zeros((l, w + 1))
    
    # Copy the data from the original array to the new array
    new_array[:, :-1] = original_array
    
    return new_array

def expanded_array_func_y(array):
    '''
    Expanding array in the vertical direction function:
        Parameters:
            array: 2d numpy array
    '''
    original_array = array
    l, w = array.shape
    # import pdb; pdb.set_trace()
    #  Create a new  array filled with zeros bigger along axis = 0
    new_array = np.zeros((l + 1, w))
    
    # Copy the data from the original array to the new array
    new_array[:-1, :] = original_array
    
    return new_array

def preprocess_labels(label, nom_classes):
    '''
    Preprocessing target labels so all within a range dependent upon the number of classes:
        Parameters:
            label: 2d numpy array
            nom_classes: integer or float value of the number of classes
    '''
    #
    
    if nom_classes == 3:
        label[label >= 2] = 1
        label[label <= -2] = -1
    
    elif nom_classes == 4:
        
        label[(label >= 2)] = 2
        label[(label <= -2)] = 2
    
    elif nom_classes == 5:
        label[label >= 2] = 2
        label[label <= -2] = -2
        
    else:
        print(' not the correct number of classes. you need class size of 3,4 or 5')
    
    return label

def values_to_count_function(nom_classes):
    '''
   Function to specify which values should be counted:
        Parameters:
            nom_classes: value of 3,4 or 5
    '''
    
    if nom_classes == 3:
        values_to_count = [-1,0,1]
    elif nom_classes == 4:
        values_to_count = [-1,0,1,2]
    elif nom_classes ==5:            
        values_to_count = [-2, -1, 0, 1, 2]
    else:
        print('Not the right number of classes to count. You need either 3,4 or 5')
    return values_to_count


#%%
def plotting_lines_and_shapes(maskx, masky, target_truth_y, target_truth_x, test_image, unwrapped, name, counting_dictionaryx, counting_dictionaryy):
    '''
    Plotting funciton:
        Parameters:
            maskx: 2d numpy array of model predicted gradients in the horizontal direction
            masky: 2d numpy array of model predicted gradients in the vertical direction
            target_truth_y: 2d numpy array of groundtruth for model predicted gradients in the vertical direction
            target_truth_x: 2d numpy array of groundtruth for model predicted gradients in the horiztonal direction
            test_image: wrapped interferogram, 2d array
            unwrapped: unwrapped interferogram, 2d array
            name: title for plots
            counting_dictionaryx: dictionary of horizontal counts
            counting_dictionary: dictionary of vertical counts
        
        output:
            counting_dictionaryx: dictionary of horizontal counts
            counting_dictionary: dictionary of vertical counts
    '''
    
    # Define the mapping of combinations to marker shapes and colors
  #  import pdb; pdb.set_trace()
  #predicted value, truth value
    mapping = {

        # (-2, 3): {'shape': 's', 'color': 'fuchsia', 'marker_size': '5', 'edge_color': (1.0, 0.0, 1.0)},
        # (-1, 3): {'shape': 's', 'color': 'fuchsia', 'marker_size': '10', 'edge_color': (1.0, 0.0, 1.0)},
        # (0, 3): {'shape': 's', 'color': 'fuchsia', 'marker_size': '10', 'edge_color':  (1.0, 0.0, 1.0)},
        # (1, 3): {'shape': 's', 'color': 'fuchsia', 'marker_size': '10', 'edge_color': (1.0, 0.0, 1.0)},
        # (2, 3): {'shape': 's', 'color': 'fuchsia', 'marker_size': '10', 'edge_color': (1.0, 0.0, 1.0)},
        # (3, 3): {'shape': 's', 'color': 'green', 'marker_size': '10', 'edge_color': (0, 1, 1)},
        
        
        (-2, -2): {'shape': 'x', 'color': 'green', 'marker_size': '5', 'edge_color': (0, 0.7, 0)},
        (-1, -2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        (0, -2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color':  (1, 0, 0)},
        (1, -2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        (2, -2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        # (3, -2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        
        
        
        (-2, -1): {'shape': 'o', 'color': 'cyan', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        (-1, -1): {'shape': 'o', 'color': 'green', 'marker_size': '5', 'edge_color': (0, 0.5, 1)},
        (0, -1): {'shape': 'o', 'color': 'cyan', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        (1, -1): {'shape': 'o', 'color': 'cyan', 'marker_size': '10', 'edge_color':(0.7, 0.7, 1)},
        (2, -1): {'shape': 'o', 'color': 'cyan', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        # (3, -1): {'shape': 'o', 'color': 'cyan', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        
        
        
        (-2, 0): {'shape': '^', 'color': 'purple', 'marker_size': '10', 'edge_color': (0, 0, 0)},
        (-1, 0): {'shape': '^', 'color': 'purple','marker_size': '10', 'edge_color':(0, 0, 0)},
        (0, 0): {'shape': '^', 'color': 'green', 'marker_size': '5' , 'edge_color': (0, 1, 0)},
        (1, 0): {'shape': '^', 'color': 'purple','marker_size': '10', 'edge_color':(0, 0, 0)},
        (2, 0): {'shape': '^', 'color': 'purple', 'marker_size': '10', 'edge_color': (0, 0, 0)},
        # (3, 0): {'shape': '^', 'color': 'purple', 'marker_size': '10', 'edge_color': (0, 0, 0)},
        
        
        (-2, 1): {'shape': 'o', 'color': 'grey', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        (-1, 1): {'shape': 'o', 'color': 'grey', 'marker_size': '10', 'edge_color':(0.7, 0.7, 1)},
        (0, 1): {'shape': 'o', 'color': 'grey', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        (1, 1): {'shape': 'o', 'color': 'green', 'marker_size': '5', 'edge_color': (0, 0.5, 1)},
        (2, 1): {'shape': 'o', 'color': 'grey', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        # (3, 1): {'shape': 'o', 'color': 'grey', 'marker_size': '10', 'edge_color': (0.7, 0.7, 1)},
        
        
        (-2, 2): {'shape': 'x', 'color': 'blue', 'marker_size': '10', 'edge_color':  (1, 0, 0)},
        (-1, 2): {'shape': 'x', 'color': 'blue', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        (0, 2): {'shape': 'x', 'color': 'blue', 'marker_size': '10', 'edge_color':  (1, 0, 0)},
        (1, 2): {'shape': 'x', 'color': 'blue', 'marker_size': '10', 'edge_color':  (1, 0, 0)},
        (2, 2): {'shape': 'x', 'color': 'green', 'marker_size': '5', 'edge_color': (0, 0.7, 0)},
        # (3, 2): {'shape': 'x', 'color': 'red', 'marker_size': '10', 'edge_color': (1, 0, 0)},
        
  
    }
    
    maskx_copied = np.copy(maskx)
    maskx_copied[maskx_copied == 2] = 4
    maskx_copied[maskx_copied == -2] = 4
    maskx_copied[maskx_copied == 1] = 4
    maskx_copied[maskx_copied == -1] = 4
   
    masky_copied = np.copy(masky)
    masky_copied[masky_copied == 2] = 4
    masky_copied[masky_copied == -2] = 4
    masky_copied[masky_copied == 1] = 4
    masky_copied[masky_copied == -1] = 4

    diff_x = maskx_copied - target_truth_x
    diff_y = masky_copied - target_truth_y
    array2 = test_image
    
    unwrapped = unwrapped
    edge_threshold = 0
    
    #identify if there are any inaccuracy in the predicted versus groundtruth gradients
    if diff_x.any() != 0:
        # import pdb; pdb.set_trace()
        any_differences = True
    elif diff_y.any() != 0:
        any_differences = True
        
    else:
        return counting_dictionaryx, counting_dictionaryy

    # Create the plot
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})

    edge_color_is_red = False  # Flag to track if any edge_color is red
    
    
    # Plot the right edges and dots
    # import pdb; pdb.set_trace()
    for i in range(maskx.shape[0]):
        for j in range(maskx.shape[1]):
            pixel_value = diff_x[i, j]
    
            # Define the edge color based on the condition
            if pixel_value != edge_threshold:
                truth_value = target_truth_x[i, j]
                predicted_value = maskx[i, j]
                edge_coloring = mapping.get((predicted_value, truth_value), (0, 0, 0))
                ax[0].plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=edge_coloring['edge_color'], linewidth=4)  # Right edge
                dot_info = mapping.get((predicted_value, truth_value))  # Default to gray x

                if edge_coloring['edge_color'] in [(0, 0.7, 0),  (1, 0, 0), (0.7, 0.7, 1), (0, 0.5, 1), (0.7, 0.7, 1),(0, 1, 0)]:
                   edge_color_is_red = True
                
                # Update the counting_dictionary
                counting_dictionaryx[(predicted_value, truth_value)] += 1

    # Plot the bottom edges and dots
    for j in range(masky.shape[0]):
        for i in range(masky.shape[1]):
            pixel_value = diff_y[i, j]
    
            if pixel_value != edge_threshold:
                truth_value = target_truth_y[i, j]
                predicted_value = masky[i, j]
                edge_coloring = mapping.get((predicted_value, truth_value), (0, 0, 0))
                # print(type(edge_coloring)) 
                ax[0].plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=edge_coloring['edge_color'], linewidth=4)  # Bottom edge
                dot_info = mapping.get((predicted_value, truth_value), {'shape': 'x', 'color': 'gray'})  # Default to gray x
                if edge_coloring['edge_color'] in [(0, 0.7, 0),  (1, 0, 0), (0.7, 0.7, 1), (0, 0.5, 1), (0.7, 0.7, 1), (0, 1, 0)]:
                   edge_color_is_red = True
               
                # Update the counting_dictionary
                counting_dictionaryy[(predicted_value, truth_value)] += 1

    
    # Add a legend to the right of the plot
    legend_labels = ['Correct Prediction 2', 'Correct Prediction 1', 'Correct Prediction 0', 'Incorrect Prediction, should be 0', 'Incorrect Prediction, should be 1','Incorrect Prediction, should be 2']
    coloring = [(0, 0.7, 0), (0, 0.5, 1), (0, 1, 0), (0, 0, 0), (0.7, 0.7, 1),  (1, 0, 0)]
    # Create a list of custom lines based on the coloring
    custom_lines = [Line2D([0], [0], color=color, lw=3) for color in coloring]
    # Add the legend with custom lines and labels
    ax[0].legend(custom_lines, legend_labels,  loc="upper left", bbox_to_anchor=(1, 1))

    # Display the images if at least one edge_color is red
    if edge_color_is_red:
        # Color NaN pixels in yellow
        cmap = plt.cm.gray
        ax[0].matshow(array2, cmap=cmap)
        ax[1].axis('off')  # Hide the axis for the colorbar
        plt.suptitle('Output and Legend')
                
        # Save the figure with tight layout to prevent cropping of labels or other content
        plt.savefig(name + ".png", bbox_inches="tight")
        plt.close()
        
        
    # Create the plot
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})

    edge_color_is_red = False  # Flag to track if any edge_color is red

    
    # Plot the right edges and dots
    for i in range(maskx.shape[0]):
        for j in range(maskx.shape[1]):
            pixel_value = diff_x[i, j]
    
            # Define the edge color based on the condition
            if pixel_value != edge_threshold:
                # import pdb; pdb.set_trace()
                truth_value = target_truth_x[i, j]
                predicted_value = maskx[i, j]
        
                edge_coloring = mapping.get((predicted_value, truth_value), (0, 0, 0))
                ax[0].plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=edge_coloring['edge_color'], linewidth=4, zorder = 1)  # Right edge

                if edge_coloring['edge_color'] in [(0, 0.7, 0),  (1, 0, 0), (0.7, 0.7, 1), (0, 0.5, 1), (0.7, 0.7, 1), (0, 1, 0)]:
                   edge_color_is_red = True

    # Plot the bottom edges and dots
    for j in range(masky.shape[0]):
        for i in range(masky.shape[1]):
            pixel_value = diff_y[i, j]
    
            if pixel_value != edge_threshold:
                # import pdb; pdb.set_trace()
                truth_value = target_truth_y[i, j]
                predicted_value = masky[i, j]
                #import pdb; pdb.set_trace()
                edge_coloring = mapping.get((predicted_value, truth_value), (0, 0, 0))
                dot_info = mapping.get((predicted_value, truth_value), {'shape': 'x', 'color': 'black'})  # Default to gray x

                ax[0].plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=edge_coloring['edge_color'], linewidth=4, zorder = 1)  # Bottom edge

                if edge_coloring['edge_color'] in [(0, 0.7, 0),  (1, 0, 0), (0.7, 0.7, 1), (0, 0.5, 1), (0.7, 0.7, 1), (0, 1, 0)]:
                   edge_color_is_red = True

    # Add a legend to the right of the plot
    legend_labels = ['Correct Prediction 2', 'Correct Prediction 1', 'Correct Prediction 0', 'Incorrect Prediction, should be 0', 'Incorrect Prediction, should be 1','Incorrect Prediction, should be 2']
    coloring = [(0, 0.7, 0), (0, 0.5, 1), (0, 1, 0), (0, 0, 0), (0.7, 0.7, 1),  (1, 0, 0)]
    
    # Create a list of custom lines based on the coloring
    custom_lines = [Line2D([0], [0], color=color, lw=3) for color in coloring]
    
    shape_labels = ['0', '2', '1']
    shape_markers = ['^', 'x', 'o']
    shape_custom_lines = [Line2D([0], [0], marker=marker, color='black', lw=0, markersize=3) for marker in shape_markers]

    # Add the legend with custom lines and labels
    # ax[0].legend(custom_lines + shape_custom_lines, legend_labels + shape_labels,  loc="upper left", bbox_to_anchor=(1, 1),title="Predicted values - Right or Wrong")
    ax[0].legend(custom_lines, legend_labels,  loc="upper left", bbox_to_anchor=(1, 1),title="Predicted values - Right or Wrong")
    
    # Display the images if at least one edge_color is red
    if edge_color_is_red:
        # Create a custom colormap with yellow for NaN values
        cmap = plt.cm.gray
        # cmap.set_bad('yellow')

        ax[0].matshow(unwrapped, cmap=cmap)
        ax[1].axis('off')  # Hide the axis for the colorbar
        plt.suptitle('Output and Legend')
                
        # Save the figure with tight layout to prevent cropping of labels or other content
        plt.savefig(name + "unw.png", bbox_inches="tight")
        plt.close()

     # Close all open figures to free up memory
    plt.close('all')
    
    return counting_dictionaryx, counting_dictionaryy
    
    
#%%
def locating_ares_of_mismatch(maskx, masky, target_truth_y, target_truth_x, test_image, unwrapped, name, counting_dictionaryx, counting_dictionaryy):
    
    # Size of the sub-arrays
    subarray_size = 64
    count = 0
    
    # Calculate the number of sub-arrays in each dimension
    num_rows = maskx.shape[0] // subarray_size
    num_cols = maskx.shape[1] // subarray_size
    
    # Iterate over the rows and columns to extract sub-arrays
    for i in range(num_rows):
        for j in range(num_cols):
            # Extract the sub-array
            maskx_sub = maskx[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]
            masky_sub = masky[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]
            target_truth_x_sub = target_truth_x[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]
            target_truth_y_sub = target_truth_y[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]
            test_imagesub = test_image[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]
            unwrappedsub = unwrapped[i * subarray_size:(i + 1) * subarray_size, j * subarray_size:(j + 1) * subarray_size]

            counting_dictionaryx, counting_dictionaryy = plotting_lines_and_shapes(maskx_sub, masky_sub, target_truth_y_sub, target_truth_x_sub, test_imagesub, unwrappedsub, name = name + str(count), counting_dictionaryx=counting_dictionaryx, counting_dictionaryy=counting_dictionaryy)
            count += 1
        
    plt.imshow(maskx)
    plt.show()
    
    plt.imshow(target_truth_x)
    plt.show()
    
    plt.imshow(masky)
    plt.show()
    
    plt.imshow(target_truth_y)
    plt.show()

    return counting_dictionaryx, counting_dictionaryy

#%%Runnign script

for each_model in range(0, len(Model_location)):
    
    #Load in the information of model, model location and number of classes to target
    folder_number = 0
    model = tensorflow.keras.models.load_model(Model_location[each_model], compile=(False))
    
    
    model.build((1,112,112,1))
    model.summary()
    from tensorflow.keras.utils import plot_model
    # import pdb; pdb.set_trace()
    # plot_model(model, to_file='model_plot_unfiltered.png', show_shapes=True, show_layer_names=True)
    # import pdb; pdb.set_trace()
    nom_classes = output_class_nom[each_model]
    model_location = model_names[each_model]
    
    counting_dictionaryx = {
        (-2, -2):0,
        (-2, 2):0,
        (-2, -1):0,
        (-2, 0):0,
        (-2, 1):0,
        # (-2,3):0,
            
        (-1, -1):0,
        (-1, -2):0,
        (-1, 0):0,
        (-1, 1):0,
        (-1, 2):0,
        # (-1,3):0,
            
        (0, 0):0,
        (0, -2):0,
        (0, -1):0,
        (0, 1): 0,
        (0, 2):0,
        # (0,3):0,
             
        (1, 1):0,
        (1, -2):0 ,   
        (1, -1):0,
        (1, 0):0,
        (1, 2):0,
        # (1,3):0,
            
        (2, 2):0,
        (2, -2):0,
        (2, -1):0,
        (2, 0):0,
        (2, 1):0,
        # (2,3):0,
        
        # (3, 2):0,
        # (3, -2):0,
        # (3, -1):0,
        # (3, 0):0,
        # (3, 1):0,
        # (3,3):0

        }

    counting_dictionaryy = {
        (-2, -2):0,
        (-2, 2):0,
        (-2, -1):0,
        (-2, 0):0,
        (-2, 1):0,
        # (-2, 3):0,
            
        (-1, -1):0,
        (-1, -2):0,
        (-1, 0):0,
        (-1, 1):0,
        (-1, 2):0,
        # (-1, 3):0,
            
        (0, 0):0,
        (0, -2):0,
        (0, -1):0,
        (0, 1): 0,
        (0, 2):0,
        # (0, 3):0,
             
        (1, 1):0,
        (1, -2):0 ,   
        (1, -1):0,
        (1, 0):0,
        (1, 2):0,
        # (1, 3):0,
            
        (2, 2):0,
        (2, -2):0,
        (2, -1):0,
        (2, 0):0,
        (2, 1):0,
        # (2, 3):0,
        
        # (3, 2):0,
        # (3, -2):0,
        # (3, -1):0,
        # (3, 0):0,
        # (3, 1):0,
        # (3, 3):0

        }


    gc.collect()
    list_dir_2 = os.listdir(file_location_base + 'wrapped/')
    counter_for_listing = 0
    for i in range(0,len(list_dir_2)):
        if counter_for_listing == 0:
            # import pdb; pdb.set_trace()
            #Loading data in (wrapped and unwrapped interferograms)
            test_image = np.load(file_location_base + 'wrapped/' + list_dir_2[i])  
            unwrapped = np.load(file_location_base + 'unwrapped/' + list_dir_2[i])
            l,w = test_image.shape
            test_image_copy = np.copy(test_image)
            
            
            # Assuming test_image is a NumPy array
            mask = np.isnan(test_image)

            plt.imshow(test_image)
            plt.show()
             
            #preparing wrapped interferogram to be put into the model
            #normalise the data
            test_image = (test_image - (- np.pi))/(((np.pi)) - (-np.pi))
            #make sure it has the correct number of channels
            array = np.expand_dims(test_image, axis = 0)
            array = np.expand_dims(array, axis = -1)
            
            #Make model prediction
            predsx, predsy = model.predict(array)
            
            #Extract the predictions and certainties in predictions
            x,y  = predsx, predsy   
            x_pred = np.argmax(x, axis = 3)
            y_pred = np.argmax(y, axis = 3)
              
            x_prob = np.max(x, axis = 3)#np.random.uniform(0.5,1, test_image.shape)#
            y_prob =np.max(y, axis = 3)# np.random.uniform(0.5,1, test_image.shape)# 
            
            mask_valuex = np.reshape(x_pred, [l,w])
            mask_valuey = np.reshape(y_pred, [l,w])  
            
            plt.imshow(mask_valuex)
            plt.colorbar()
            plt.show()
            
            most_probable_probx = np.reshape(x_prob, [l,w])
            most_probable_proby = np.reshape(y_prob, [l,w])  
            
            
            #Get truth dataset
            array = test_image_copy
            arrayx = expanded_array_func_x(array)
            arrayy = expanded_array_func_y(array)
            unwrapx = expanded_array_func_x(unwrapped)
            unwrapy = expanded_array_func_y(unwrapped)

            k_valuesx = np.round((unwrapx- arrayx)/(2*np.pi))
            k_valuesy = np.round((unwrapy - arrayy)/(2*np.pi))
            
            k_valuesx[:, -1] = k_valuesx[:, -2]
            k_valuesy[-1, :] = k_valuesy[-2, :]
           
            label_hor = np.round(np.diff(k_valuesx, axis = 1))
            label_vert = np.round(np.diff(k_valuesy, axis = 0))
           
            #  Replace NaN values with 0
            label_hor[np.isnan(label_hor)] = 0
            label_vert[np.isnan(label_vert)] = 0
    
            target_truth_x = preprocess_labels(label_hor, nom_classes)
            target_truth_y = preprocess_labels(label_vert, nom_classes)

            if nom_classes == 5:            
                mask_valuex = mask_valuex -2
                mask_valuey = mask_valuey -2
                x = x-2
                y = y-2
           
            elif nom_classes == 4:   
                # import pdb; pdb.set_trace()
                mask_valuex = mask_valuex -1
                mask_valuey = mask_valuey -1
                x = x-1
                y = y-1
           
            elif  nom_classes == 3:
                mask_valuex = mask_valuex -1
                mask_valuey = mask_valuey -1
                x = x-1
                y = y-1
                
            #Now i want to look at occasions where the prediction is wrong
            target_truth_x_flattened = target_truth_x.flatten()
            mask_valuex_flattened = mask_valuex.flatten()
            target_truth_y_flattened = target_truth_y.flatten()
            mask_valuey_flattened = mask_valuey.flatten()
            
            # mask_valuec =  np.load(file_location_base + 'deformation_mask/' + list_dir[j] + '/' + list_dir_2[i])
            mask_valuec = mask
            #NOw lets see if locations are within the noise regions
            target_truth_x_flattened = (target_truth_x * mask_valuec).flatten()
            mask_valuex_flattened = (mask_valuex * mask_valuec).flatten()
            target_truth_y_flattened = (target_truth_y * mask_valuec).flatten()
            mask_valuey_flattened = (mask_valuey * mask_valuec).flatten()
            
            #Confusion matrix for x and y directions
            confusion_matx = confusion_matrix(target_truth_x_flattened, mask_valuex_flattened)
            confusion_maty = confusion_matrix(target_truth_y_flattened, mask_valuey_flattened)

            x_match = target_truth_x_flattened == mask_valuex_flattened
            y_match = target_truth_y_flattened == mask_valuey_flattened
            
            x_match_shape = np.reshape(x_match, (mask_valuex.shape))
            y_match_shape = np.reshape(y_match, (mask_valuey.shape))
            
            most_probx_copy = np.copy(most_probable_probx)
            most_proby_copy = np.copy(most_probable_proby)
            
            #Highlight locations where probabilties are greater than 0.8 and make 1
            most_probable_probx[most_probable_probx >= 0.1] = 1
            most_probable_probx[most_probable_probx < 0.1] = 0
            
            most_probable_proby[most_probable_proby >= 0.1] = 1
            most_probable_proby[most_probable_proby < 0.1] = 0
    
            mask_valuex = mask_valuex.astype('float32')
            mask_valuey = mask_valuey.astype('float32')
            target_truth_x = target_truth_x.astype('float32')
            target_truth_y = target_truth_y.astype('float32')          
            
            #Make 0 locations where wrong but probability is less than 0.8
            mask_valuex = mask_valuex# * most_probable_probx
            mask_valuey = mask_valuey# * most_probable_proby 
            target_truth_x = target_truth_x # * most_probable_probx
            target_truth_y = target_truth_y #* most_probable_proby

            name = output_file_location + model_location + '/'
            
            if not os.path.exists(name):
                os.mkdir(name)
           
            name = name + '/' 
            
            if not os.path.exists(name):
                os.mkdir(name)
                
            name = name + str(folder_number) + '/'
            if not os.path.exists(name):
                os.mkdir(name)
            
            
            counting_dictionaryx, counting_dictionaryy = locating_ares_of_mismatch(mask_valuex, mask_valuey, target_truth_y, target_truth_x, test_image, unwrapped, name = name, counting_dictionaryx=counting_dictionaryx, counting_dictionaryy=counting_dictionaryy)
            folder_number += 1
            counter_for_listing = 0
           # import pdb; pdb.set_trace()
            del test_image, unwrapped, array, predsx, predsy
            del x, y, x_pred, y_pred, x_prob, y_prob, mask_valuex, mask_valuey, most_probable_probx, most_probable_proby
            del arrayx, arrayy, unwrapx, unwrapy, k_valuesx, k_valuesy, label_hor, label_vert, target_truth_x, target_truth_y
            del mask_valuec, target_truth_x_flattened, mask_valuex_flattened, target_truth_y_flattened, mask_valuey_flattened
            del confusion_matx, confusion_maty, x_match, y_match, x_match_shape, y_match_shape, most_probx_copy, most_proby_copy

        else:
            counter_for_listing += 1

            
            
            