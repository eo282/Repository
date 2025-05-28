# Eilishs_repository

The code provided in this repository is part of ongoing research investigating how to improve phase unwrapping undertaken by Eilish O'Grady at the University of Leeds
The main_v2.py code allows the training of a U-Net model for a 3 class semantic segmentation model.
Files required:
Inputs:
2d arrays of channel 1. this code expects values from -pi (minimum value) to pi (maximum value) but you can edit this in the normalisation step in the generator to a different value range
Output:
Target label of the same size as input where each pixel is 1 of 3 classes

anlaysing_model_output.py is used for visulisation of where the model is correctly and incorrectly identify pixel classification

To create environment for this code, please use the cuda_active_env.yml file

If using please cite this github repository
