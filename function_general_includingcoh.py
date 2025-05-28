#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:37:15 2024

@author: eeerog
file containing functions related to the processing of the data for model training
preparing data for a 3 class semantic segmentation task unet model
input is a 2d array of channel 1, output is 2d array of same size channel = 1
"""
#%%import modules

import numpy as np
import os
import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import tensorflow

#%%procces file
def process_file(file_path, sub_folder, targets_dir, inputs_dir, class_nom,array_size_indiv):

    variable_name = os.path.relpath(file_path, targets_dir)
    targets_out = np.load(os.path.join(targets_dir + '/', variable_name)).astype(np.float32)
    inputs_in = np.load(os.path.join(inputs_dir  + '/', variable_name)).astype(np.float32)

    inputs_in = inputs_in[:array_size_indiv, :array_size_indiv]
    targets_out = targets_out[:array_size_indiv, :array_size_indiv]
    mask = np.ones(targets_out.shape)
    mask[np.isnan(inputs_in)] = 0
    mask[inputs_in == 0] = 0
    # plt.imshow(inputs_in)
    # plt.show()
    # Check for NaN values and neighbors
    nan_indices = np.isnan(inputs_in)
    neighbor_indices = (
        np.roll(inputs_in, 1, axis=0),
        np.roll(inputs_in, -1, axis=0),
        np.roll(inputs_in, 1, axis=1),
        np.roll(inputs_in, -1, axis=1)
    )

    # Set mask to 0 where there are NaN values or at neighboring pixels
    mask[nan_indices | np.any(np.isnan(neighbor_indices), axis=0)] = 0
    
    #ensure target data is given correct target values dependent upon the number of classes required for training
    if class_nom == 3:   
        targets_out[np.isnan(targets_out)] = 0
        targets_out[targets_out >= 2] = 1
        targets_out[targets_out <= -2] = -1
        targets_out[mask == 0] = 0
    
    else:
	print('you can only have a 3 class model')
    

        
    return targets_out, mask, sub_folder


def process_file_parallel(args):
    file_path, sub_folder,  target_dir, inputs_dir, input_nom, array_size_indiv = args
    return process_file(file_path, sub_folder, target_dir, inputs_dir, input_nom, array_size_indiv)

def listing_function(inputs, class_nom, array_size_indiv):
    target_dir = inputs + 'x_grad/'
    inputs_dir = inputs + 'wrapped_noised/'
    # import pdb; pdb.set_trace()
    folder_dir = os.listdir(inputs_w) 
    array_target = []
    array_c = []
    array_coh = []

    for each_folder in folder_dir:
        file_paths_v =glob.glob(os.path.join(target_dir + each_folder + '/', '*.npy'), recursive=True)
        
        # Define the number of processes (adjust as needed)
        num_processes = 10
        # Create a pool of worker processes
        pool = Pool(num_processes)
        # Prepare the arguments for parallel processing
        args = [(file_path,each_folder, target_dir, inputs_dir, class_nom, array_size_indiv) for file_path in file_paths_v]
        # Use the pool to parallelize the processing of files
        results = pool.map(process_file_parallel, args)
        # Close the pool to release resources
        pool.close()
        pool.join()

        for targets_out, mask2, sub_folder in results:
            array_target.append(targets_out)

    # Calculate class frequencies from all the data together
    # import pdb; pdb.set_trace()
    all_file_h = np.concatenate(array_target)
    
    #calculate the weighting
    if class_nom == 3:   
        class_counts = [np.count_nonzero(all_file_h == -1), np.count_nonzero(all_file_h == 0)  , np.count_nonzero(all_file_h == 1) , np.count_nonzero(all_file_h == 2) ]
        total_samples = sum(class_counts)
        class_frequencies = [count / total_samples for count in class_counts]

        # Rest of your code to calculate class weights goes here
        
        class_weights = [1 - count  for count in class_frequencies]
	    
	# normalising part
        sum_weights = sum(class_weights)
        
        # Normalize by dividing each weight by the sum of weights (to ensure they sum to 1)
        normalized_class_weights = [weight / sum_weights for weight in class_weights]
       
        num_classes = class_nom
        total_samples_m1 = total_samples/(num_classes * class_counts[0])
        total_samples_0 = total_samples/(num_classes * class_counts[1])
        total_samples_1 = total_samples/(num_classes * class_counts[2])
        
        normalized_class_weights = [total_samples_m1, total_samples_0, total_samples_1]
    else:
	print('you can only use this for a 3 class classification problem')
       
        
    return (
        normalized_class_weights
    )

#%%Prepare data subsection
def prepare_data_subsection(inputs):
    """
    To be used as part of prepare data
    Checks the directory structure type
    Performs task of creating list of data locations
    Inputs:
    inputs: directory location
    Return:
    list_name: list of files
    """

    import numpy as np
    import os
    import pandas as pd

    list_name = []
	
    # import pdb; pdb.set_trace()
    input_dir = os.listdir(inputs)
    # import pdb; pdb.set_trace()
    for i in input_dir:
        # import pdb; pdb.set_trace()
        if i.endswith(".npy"):
            new_path = os.path.join(inputs + i)
            list_name.append(new_path)
        else:
            new_inputs_dir = os.path.join(inputs + i)
            directory = os.listdir(new_inputs_dir)
            
            # import pdb; pdb.set_trace()
            
            
            for q in directory:
                if q.endswith(".npy"):
                    for k in range(0, len(directory)):
                        if directory[k].endswith(".npy"):
                            new_path = os.path.join(
                                new_inputs_dir + '/' + directory[k])
                            list_name.append(new_path)

    return list_name

#%%prepare data multi output with coherence output too
def prepare_data_multioutput(input_directory, target_directory,
                 csv_directory = None, csv_file_names = 'semi_supervised_model',percentage_validation = 0.2):
   
    """
    Module for preparing data for training with semi-supervised model
    
    Inputs:
    input_directory: location of input training data
    target_directory: location of targets for input training data
    mask_directory: location of masks for input training data, set as None
    deformation_directory: location of deformation masks to match deformation masks to match input training data, set as None
    csv_directory: location to save csv files to
    csv_file_names: giving new names to csv file, set to semi_supervised_model
    percentage_validation: the percentage of training data to make validation data
    """

    
     #use prepare data subsection to get a list of all files avaliable for training
    # import pdb; pdb.set_trace()
    training_inputs = prepare_data_subsection(input_directory)  #training inputs
    training_targets = prepare_data_subsection(target_directory)    #target outputs in the horizontal direction
   
    training_inputs.sort()  #sort so in alphabetical order
    training_targets.sort() #sort so in alphabetical order
   
    training_inputsr = []
    training_inputsi = []
    training_targets2 = []

    data = pd.DataFrame({
	    'Inputs': training_inputs,
	    'Targets': training_targets
	})
            
    
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=percentage_validation, shuffle = True)
    
    training_dataframes = train_data
    validation_dataframes = val_data
    
    if not os.path.exists(csv_directory):
        os.mkdir(csv_directory)
    nst = len(training_dataframes)
    nsv = len(validation_dataframes)
    
    #create varieables of the location of the csv files to output
    training_data_csv_list = csv_directory + csv_file_names + "t.csv"
    validation_data_csv_list = csv_directory + csv_file_names + "v.csv"
    
    training_dataframes.to_csv(training_data_csv_list, index = 'False')
    validation_dataframes.to_csv(validation_data_csv_list, index = 'False')

    return training_data_csv_list, validation_data_csv_list, nst, nsv



#%%Generator

def preprocess_labels(label, class_nom):
    '''
    Function for preprocessing labels in generator during training
    ensure 3 classes - all values above 1 = 1, all values less than -1 = -1
    Inputs:
    label: 2d array containing target labels
    class_nom: number of classes for classification (int/float)
    return:
    label: preprocessed 2d array target label
    '''
    import numpy as np    
    # Replace specific values according to number of classes
    
    if class_nom ==3:
        label[label >= 2] = 1
        label[label <= -2] = -1
        
        label[np.isnan(label)] = 0
        label = label + 1

    else:
        print('error - class number needs to be 3')

    return label


def add_noise_with_coherence(array, target_coherence):
    '''
    Function to add noise to 2d array based on a target coherence
    Inputs:
    array: 2d array
    target_coherence: coherence value between 0.0 and 1.0
    Return:
    noisy_array: 2d array noised according to coherence value
    '''
    # Calculate the noise level based on the target coherence
    noise_level = np.sqrt(1 - target_coherence)

    # Add Gaussian noise to the array
    # import pdb; pdb.set_trace()
    data_std = np.std(array)*noise_level
    noise = np.random.normal(0, data_std, size=array.shape)
    noisy_array = array + noise#(noise_level * np.random.normal(size=array.shape))
    # noisy_array = np.mod(noisy_array, 2 * np.pi)
    noisy_array = np.clip(noisy_array, -np.pi, np.pi)

    return noisy_array



def samples_gen_part_two_output_coh(samples, batch_size, shuffle_data, array_size = (112, 112, 1), class_nom = 3):
    """
    function for preparing batches for training
    multihead output, no cohereence
    
    Inputs:
        samples: sample csv
        batch_size: batch size required as output
        shuffle_data: to allow data to be shuffled to ensure randomness in training dataset
        array_size = output array size per training data
        addition: to determine if phase gradient wanted (addition = 0) or ambiguity number (addition = o/w)
    
    Outputs:
        X_train: training dataset of input training data
        y_train_hor, y_train_vert, coh_train: dataset of targets
        mask_train, defo_train: masks
    """
    
    import numpy as np
    import math
    import random
    import pandas as pd
    from PIL import Image
    while True:

    #number samples = length samples
    #ensure batch size is an integer
        num_samples = len(samples)
        batch_size = int(batch_size)
        np.random.shuffle(samples)
        #preparing batches
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            #making output dictionaries
            X_train = []
            y_target = []

            for batch_sample in batch_samples:
                
                batch_sample_length = len(batch_sample)
                input_file_name = batch_sample[0]
                target_file_name = batch_sample[1]
                                
                input_file = np.load(input_file_name)
                label_target = np.load(target_file_name)

                label_target[input_file == 0] = np.nan
		label_target[np.isnan(input_file)] = np.nan

		label_target = preprocess_labels(label_target, class_nom)

		random_values = np.random.uniform(-np.pi, np.pi, size=input_file.shape)
                input_file[np.isnan(input_file)] = random_values[np.isnan(input_file)]               
                
                coht = np.random.uniform(0.3, 0.9999)
                coht = round(coht, 2)
                input_file = add_noise_with_coherence(input_file, coht)

                input_file = (input_file - (-np.pi)) / ((np.pi) - (-np.pi))#
                y_target = np.append(y_target, label_target)
                X_train = np.append(X_train, input_file)
            
            X_train = np.array(X_train)
            y_target = np.array(y_target)

            h = array_size[0]
            w = array_size[1]
            
            X_train = np.resize(X_train, (batch_size ,h,w,1))
            y_target = np.resize(y_target, (batch_size ,h,w,1))
		
            # import pdb; pdb.set_trace()
            yield X_train, y_target

X_train = []
y_train_hor = []
y_train_vert = []
y_train_coh = []


#%%Model lets go - multi output with coherence

def load_samples(csv_file, class_nom):
    """
    Function to load sample locations into list
    
    Input:
        csv_file: csv file location
        
    Output:
        samples: list of locations of each of inputs, targets, (optional: masks, deformations)
    """
    #import required modules
    import pandas as pd
    import numpy as np
    # import pdb; pdb.set_trace()
    #read csv
    data = pd.read_csv(csv_file)
    data_keys_length = len(data.keys()) - 1
    
    if data_keys_length == 3:
        data = data[['Inputs_w','Targets_Hor']]
        file_namesr = list(data.iloc[:,0])
        labels_hor = list(data.iloc[:,1])
    
        samples = []
        for sampr, lab_h in zip(file_namesr, labels_hor):
            samples.append([sampr,lab_h])
	return samples
    
    else:
        print('there is a problem with the way the data is being loaded - you should only have 1 input and 1 target')
    
    	return None

def Lets_go_model_multi_coh(input_directory, target_directory, csv_directory = None, csv_file_names = 'training_files_locations',
                 percentage_validation = 0.2, batch_size = 20, array_size = (112, 112, 1),
                 addition = 0, shuffle_data = True, class_nom = 3):
'''
Function to pre-prepare training dataset
Inputs:
input_directory_synw: file location directory of wrapped input data
target_directory_syn_hor: file location directory of horizontal target data
target_directory_syn_vert: file direcotry of vertical target data
csv_directory: location to save csv files containing list of file locations
csv_file_names: name to save csv files under
percentage_validation: value between 0 and 1 to seperate the data into training and validation dataset
batch_size: batch size (int)
array_size: size of wrapped data, 3 values (h,w,c)
addition: automatically set to 0
shuffle_data: true or false
class_nom: number of classes for training. can be 3,4 or 5
unwrapped_directory: file location of unwrapped data
Returns:
train_samples_syn: file locations for input, targets for training
valid_samples_syn: file locations for input, targets for validation
nst: number of training samples
nsv: number of validation samples
'''
    
    # import pdb; pdb.set_trace()
    training_data_syn, validation_data_syn, nst, nsv = prepare_data_multioutput(input_directory = input_directory, 
                                                                                    target_directory = target_directory,
                                                                                    csv_directory = csv_directory, csv_file_names = csv_file_names,
                                                                                    percentage_validation = percentage_validation)
   # import pdb; pdb.set_trace()
    train_samples_syn = load_samples(training_data_syn, class_nom)
    valid_samples_syn = load_samples(validation_data_syn, class_nom)
    
    print('length training samples' + str(len(train_samples_syn)))
    print('length validation samples' + str(len(valid_samples_syn)))
        
    return train_samples_syn, valid_samples_syn,  nst, nsv


