#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:19:22 2023

@author: eeerog
File containing functions and run code to patch data ready for model training
reliant upon data size being divisable by partition length
files need to be organised as:
overall folder:
wrapped/unwrapped etc.
coherence folders
files
"""

#%%Importing modules
import numpy as np
import os
from skimage.util import view_as_blocks
from numpy import save
import matplotlib.pyplot as plt

#%%General Functions
def checking_if_location_exist(location):
'''
Function to check location exists, if doesn't exist, make location
Input:
location: string, directory to check whether exists
'''

    if not os.path.exists(location):
        os.mkdir(location)

def expanded_array_func_x(array, patch_size):
'''
Function to expand 2d array in the horizontal direction
Input:
array: 2d array of equal width and height
patch_size: size of width
return:
new_array: expanded 2d array containing same values apart from extra column which is a duplicate of the previous column
'''
    original_array = array
    
    # Create a new zeros array of size patch_Size, patch_size + 1
    new_array = np.zeros((patch_size, patch_size + 1))
    
    # Copy the data from the original array to the new array
    new_array[:, :-1] = original_array
    new_array[:, -1] = new_array[:, -2]
    
    return new_array


def preprocess_image(image_scaled):

'''
Function to preprocess image
input:
image_scaled: 2d array of values -pi to pi
return:
image_scaled: normalised array with values between 0 and 1
mask: 2d array showing locations of nans in input image_Scaled, nans given value 0, otherwise 1
'''
    import numpy as np
    
    mask = np.ones(image_scaled.shape)
    mask2 = np.ones(image_scaled.shape)
    mask2[np.isnan(image_scaled)] = 0
    
    # Check for NaN values and neighbors
    nan_indices = np.isnan(image_scaled)
    neighbor_indices = (
        np.roll(image_scaled, 1, axis=0),
        np.roll(image_scaled, -1, axis=0),
        np.roll(image_scaled, 1, axis=1),
        np.roll(image_scaled, -1, axis=1)
    )
    
    # Set mask to 0 where there are NaN values or at neighboring pixels
    mask[nan_indices | np.any(np.isnan(neighbor_indices), axis=0)] = 0
    #replace nans in image_scaled with random values between -pi and pi
    random_values = np.random.uniform(-np.pi, np.pi, size=image_scaled.shape)
    image_scaled[mask == 0] = random_values[mask == 0]
    #normalise all values in image_scaled to be between 0 and 1
    image_scaled = (image_scaled - (- np.pi))/(((np.pi)) - (-np.pi))
    return image_scaled, mask

def patching_dataset(data_file_locations, saving_file_location, partition_length):
    """
    Going to try patching dataset to allow to load in smaller images and deal with class imbalance
    Input:
    data_file_locations: directory of data file locations - folder above specific wrapped, unwrapped etc. data. within folders have seperate coherence folders
    saving_file_locations: base location to save files - folder above specific wrapped, unwrapped etc. data
    partition_length: size of patches the big 2d array is going to be split into
    return:
    saved files for wrapped data, unwrapped data, x gradient data, y gradient data, unwrapped data, deformation mask data
    """
    
    #setting location names
    home_location = data_file_locations
    saving_location = saving_file_location
    partition_length = partition_length
    
    #define the wrapped location
    wrapped_location = home_location + '/wrapped/'
    #list the directory - should return a list of coherence level folders
    coh_levels = os.listdir(wrapped_location)
    
    
    #loop through coherence levels to make location variables for wrapped, unwrapped, x grad, y grad, mask
    for coh_lib in range(0, len(coh_levels)):
        wrapped_location_onwards = wrapped_location + coh_levels[coh_lib]
        unwrapped_location = home_location + '/unwrapped/'  + coh_levels[coh_lib]
        xgrad_location = home_location + '/x_grad/'  + coh_levels[coh_lib]
        mask_location = home_location + '/deformation_mask/'  + coh_levels[coh_lib]
        
        #masking the saving locations for wrapped, unwrapped, xgrad, y grad, mask patches
        wrapped_location_onwards_saving = saving_location + '/wrapped/' + coh_levels[coh_lib] + '/'
        unwrapped_location_saving = saving_location + '/unwrapped/'  + coh_levels[coh_lib] + '/'
        xgrad_location_saving = saving_location + '/x_grad/'  + coh_levels[coh_lib] + '/'
        mask_location_saving = saving_location + '/deformation_mask/'  + coh_levels[coh_lib] + '/'
        
        checking_if_location_exist(wrapped_location_onwards_saving)
        checking_if_location_exist(unwrapped_location_saving)
        checking_if_location_exist(xgrad_location_saving)
        checking_if_location_exist(mask_location_saving)
        
        wrapped_location_onwards_list = os.listdir(wrapped_location_onwards)

        for each_file in wrapped_location_onwards_list:
            
            #load each file - will match up due to way files are being loaded
            wrapped = np.load(wrapped_location_onwards + '/' + each_file)
            unwrapped = np.load(unwrapped_location + '/' + each_file)
            xgrad = np.load(xgrad_location + '/' + each_file)
            mask = np.load(mask_location + '/' + each_file)
            
            #i want to look at the mask and the wrapped data mainly
            #First need to patch - preparing patches

    
            patch_size = (partition_length, partition_length)
            patches_wrapped = view_as_blocks(wrapped, block_shape=patch_size).squeeze()
            patches_unwrapped = view_as_blocks(unwrapped, block_shape=patch_size).squeeze()
            patches_xgrad = view_as_blocks(xgrad, block_shape=patch_size).squeeze()
            patches_mask = view_as_blocks(mask, block_shape=patch_size).squeeze()
            
            count = 0
            #saving on a patch by patch basis
            for i in range(patches_wrapped.shape[0]):
                for j in range(patches_wrapped.shape[1]):
                    
                    patch_wrapped = patches_wrapped[i, j]
                    patch_unwrapped = patches_unwrapped[i,j]
                    patch_xgrad = patches_xgrad[i,j]
                    patch_mask = patches_mask[i,j]
                    
                    countpw = np.count_nonzero(np.isnan(patch_wrapped))
                    patch_size_total = partition_length * partition_length
                    percentage_nan = (countpw / patch_size_total)*100
                    
                    
                    if percentage_nan >= 80:
                        #means likely a noise patch - don't want to augment patches with lots of noise
                        save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', patch_wrapped)
                        save(unwrapped_location_saving + each_file + str(count) + '.npy', patch_unwrapped)
                        save(xgrad_location_saving + each_file + str(count) + '.npy', patch_xgrad)
                        save(mask_location_saving + each_file + str(count) + '.npy', patch_mask)
                        
                        count += 1
                        
                    else:
                        #means its likely not a noise patch
                        #see if it has a lot of +1, -1  or more - so minority class for data augmentation
                        copypw = np.copy(patch_wrapped)
                        copypw, mask = preprocess_image(copypw)

                        xgrad_extralocs = np.count_nonzero(patch_xgrad*mask >= +1) + np.count_nonzero(patch_xgrad*mask <= -1) 
                        ygrad_extralocs = np.count_nonzero(patch_ygrad*mask >= +1) + np.count_nonzero(patch_ygrad*mask <= -1)
                        
                        per_xextremes = (xgrad_extralocs/patch_size_total)*100
                        per_yextremes = (ygrad_extralocs/patch_size_total)*100
                        comb_per = per_yextremes + per_xextremes
                        
                        if per_xextremes >= 30 or per_yextremes >= 30 or comb_per >= 30:
                            
                            #where above 30% of minority sample within patch augment data
                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', patch_wrapped)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', patch_unwrapped)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', patch_xgrad)
                            save(mask_location_saving + each_file + str(count) + '.npy', patch_mask)  
                            count += 1
                            
                            #rotate 90oc
                            rot_wrapped = np.rot90(patch_wrapped, k=1)
                            rot_unwrapped = np.rot90(patch_unwrapped, k=1)
                            rot_mask = np.rot90(patch_mask, k=1)
         
                            arrayx = expanded_array_func_x(rot_wrapped, partition_length)
                            arrayy = expanded_array_func_y(rot_wrapped, partition_length)
                            unwrapx = expanded_array_func_x(rot_unwrapped, partition_length)

                            k_valuesx = (unwrapx- arrayx)/(2*np.pi)
                            
                            label_hor = np.round(np.diff(k_valuesx, axis = 1))
                            
                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', rot_wrapped)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', rot_unwrapped)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', label_hor)
                            save(mask_location_saving + each_file + str(count) + '.npy', rot_mask)  
                            count += 1
                            
                            #rotate 180oc
                            rot_wrapped = np.rot90(patch_wrapped, k=2)
                            rot_unwrapped = np.rot90(patch_unwrapped, k=2)
                            rot_mask = np.rot90(patch_mask, k=2)
         
                            arrayx = expanded_array_func_x(rot_wrapped, partition_length)
                            arrayy = expanded_array_func_y(rot_wrapped, partition_length)
                            unwrapx = expanded_array_func_x(rot_unwrapped, partition_length)

                            k_valuesx = (unwrapx- arrayx)/(2*np.pi)
                            
                            label_hor = np.round(np.diff(k_valuesx, axis = 1))

                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', rot_wrapped)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', rot_unwrapped)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', label_hor)
                            save(mask_location_saving + each_file + str(count) + '.npy', rot_mask)  
                            count += 1
                            
                            #rotate 270oc
                            rot_wrapped = np.rot90(patch_wrapped, k=3)
                            rot_unwrapped = np.rot90(patch_unwrapped, k=3)
                            rot_mask = np.rot90(patch_mask, k=3)
                            
                            arrayx = expanded_array_func_x(rot_wrapped, partition_length)
                            arrayy = expanded_array_func_y(rot_wrapped, partition_length)
                            unwrapx = expanded_array_func_x(rot_unwrapped, partition_length)

                            k_valuesx = (unwrapx- arrayx)/(2*np.pi)
                            label_hor = np.round(np.diff(k_valuesx, axis = 1))
   
                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', rot_wrapped)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', rot_unwrapped)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', label_hor)
                            save(mask_location_saving + each_file + str(count) + '.npy', rot_mask)  
                            count += 1   
                            
                            # Assuming your 2D array is named 'data'
                            #flip data in both vertical and horizontal direction
                            flipped_both_w = np.flip(patch_wrapped, axis=(0, 1))
                            flipped_both_uw = np.flip(patch_unwrapped, axis=(0, 1))
                            flipped_both_m = np.flip(patch_mask, axis = (0,1))
                            
                            cflipped_both_w = np.copy(flipped_both_w)
                            cflipped_both_uw = np.copy(flipped_both_uw)
                            cflipped_both_m = np.copy(flipped_both_m)
                            
                            arrayx = expanded_array_func_x(flipped_both_w, partition_length)
                            arrayy = expanded_array_func_y(flipped_both_w, partition_length)
                            unwrapx = expanded_array_func_x(cflipped_both_uw, partition_length)

                            k_valuesx = (unwrapx- arrayx)/(2*np.pi)
                            
                            label_hor = np.round(np.diff(k_valuesx, axis = 1))

                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', cflipped_both_w)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', cflipped_both_uw)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', label_hor)
                            save(mask_location_saving + each_file + str(count) + '.npy', cflipped_both_m)  
                            count += 1
                            
                            
                            # Assuming your 2D array is named 'data'
				#flips data in the vertical direction only
                            flipped_both_w = np.flip(patch_wrapped, axis=0)
                            flipped_both_uw = np.flip(patch_unwrapped, axis=0)
                            flipped_both_m = np.flip(patch_mask, axis=0)
                            
                            cflipped_both_w = np.copy(flipped_both_w)
                            cflipped_both_uw = np.copy(flipped_both_uw)
                            cflipped_both_m = np.copy(flipped_both_m)
                            
                            arrayx = expanded_array_func_x(flipped_both_w, partition_length)
                            arrayy = expanded_array_func_y(flipped_both_w, partition_length)
                            unwrapx = expanded_array_func_x(cflipped_both_uw, partition_length)

                            k_valuesx = (unwrapx- arrayx)/(2*np.pi)

                            label_hor = np.round(np.diff(k_valuesx, axis = 1))

                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', cflipped_both_w)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', cflipped_both_uw)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', label_hor)
                            save(mask_location_saving + each_file + str(count) + '.npy', cflipped_both_m)  
                            count += 1  
                            
                        else:
			     #if not a high percentage of minority classes, save only once
                            save(wrapped_location_onwards_saving + each_file + str(count) + '.npy', patch_wrapped)
                            save(unwrapped_location_saving + each_file + str(count) + '.npy', patch_unwrapped)
                            save(xgrad_location_saving + each_file + str(count) + '.npy', patch_xgrad)
                            save(mask_location_saving + each_file + str(count) + '.npy', patch_mask)
                            
                            count += 1
                                                    
#%%Run the code above
       
data_file_locations = '/home/eeerog/syinterferopy/SyInterferoPy/trainingdataset_v2/ambiguity_gradient'                
saving_file_location =  '/home/eeerog/PhD_3rd_year/patches_larger/'         
partition_length = 112        
                    
                    
patching_dataset(data_file_locations, saving_file_location, partition_length)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
