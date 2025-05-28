# Eilishs_repository

The code provided in this repository is part of ongoing research investigating how to improve phase unwrapping undertaken by Eilish O'Grady at the University of Leeds
The main_v2.py code allows the training of a U-Net model
anlaysing_model_output.py is used for visulisation of where the model is correctly and incorrectly identifying phase gradients
patching_dataset is used for cutting up large interferograms into smaller size interferogram patches and augmenting the data for where the minority classes are present

For the purposes of this github repository, it provides code to train a UNet capable of classifying gradients into 3 classes

To create environment for this code, please use the cuda_active_env.yml file

If using please cite this github repository
