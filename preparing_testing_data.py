#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:17:06 2025

@author: eeerog
File to prepare training dataset by cutting interferograms into smaller patches
"""

#%%Variables
import tifffile
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy import interpolate
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import generic_filter
from scipy import stats
import scipy.signal as sps
import scipy.linalg as spl
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft, convolve
from sklearn.linear_model import HuberRegressor
from skimage.util import view_as_blocks


location_base = '/home/eeerog/folder_to_end_all_folders/training_data_bulkingnumbers//'
saving_path = '/home/eeerog/folder_to_end_all_folders/training_data_bulkingnumbers/patched_64/'
wrapped = location_base + 'wrapped/'
unwrapped = location_base + 'unwrapped/'
partition_length =  64
#%%functions
def expanded_array_func_x(array, patch_size):
    # Create a 223x223 array (replace this with your actual data)
   
    original_array = array

    
    # Create a new 224x225 array filled with zeros
    new_array = np.zeros((patch_size, patch_size + 1))
    
    # Copy the data from the original array to the new array
    new_array[:, :-1] = original_array
    new_array[:, -1] = new_array[:, -2]
    
    return new_array

def expanded_array_func_y(array, patch_size):
    original_array = array

    # Create a new 224x225 array filled with zeros
    new_array = np.zeros((patch_size + 1, patch_size))
    
    # Copy the data from the original array to the new array
    new_array[:-1, :] = original_array
    new_array[-1, :] = new_array[-2, :]
    
    return new_array

def checking_if_location_exist(location):
    if not os.path.exists(location):
        os.mkdir(location)
        
        
def get_ambiguity_and_gradients(saving_path, phase_after_s, phase_orig_s, phase_unwrap_s, phase_noise_s, minimum, filtered_phase, coherence):
    
    '''
    Inputs:
       wrapped_dir | location wrapped interferograms original size saved
       unwrapped_dir | location unwrapped interferograms original size saved
       saving_path | base locations for interferograms original size -1 x original size -1 to be saved too
       coherance | coherance value for folder name of interferogram
    '''
    
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import math
    from numpy import save
    import numpy.ma as ma
    import random
    import struct
    
    # import pdb; pdb.set_trace()
    count = minimum
    
                
    checking_if_location_exist(saving_path)
    checking_if_location_exist(saving_path  + '/wrapped')
    checking_if_location_exist(saving_path + '/wrapped_noised')
    checking_if_location_exist(saving_path + '/wrapped_filtered')
    checking_if_location_exist(saving_path + '/unwrapped')
    checking_if_location_exist(saving_path + '/x_grad')
    checking_if_location_exist(saving_path + '/y_grad')
    checking_if_location_exist(saving_path + '/unwrapped_noised')
    checking_if_location_exist(saving_path + '/x_grad_noised')
    checking_if_location_exist(saving_path + '/y_grad_noised')
    checking_if_location_exist(saving_path + '/unwrapped_filtered')
    checking_if_location_exist(saving_path + '/x_grad_filtered')
    checking_if_location_exist(saving_path + '/y_grad_filtered')
    
    checking_if_location_exist(saving_path  + '/wrapped/' + str(coherence))
    checking_if_location_exist(saving_path + '/wrapped_noised/' + str(coherence))
    checking_if_location_exist(saving_path  + '/wrapped_filtered/' + str(coherence))
    checking_if_location_exist(saving_path + '/unwrapped/' + str(coherence))
    checking_if_location_exist(saving_path + '/x_grad/' + str(coherence))
    checking_if_location_exist(saving_path + '/y_grad/' + str(coherence))
    checking_if_location_exist(saving_path + '/unwrapped_noised/' + str(coherence))
    checking_if_location_exist(saving_path + '/x_grad_noised/' + str(coherence))
    checking_if_location_exist(saving_path + '/y_grad_noised/' + str(coherence))
    checking_if_location_exist(saving_path + '/unwrapped_filtered/' + str(coherence))
    checking_if_location_exist(saving_path + '/x_grad_filtered/' + str(coherence))
    checking_if_location_exist(saving_path + '/y_grad_filtered/' + str(coherence))


    wrapped_location_saved = saving_path + 'wrapped/' + str(coherence)
    unwrapped_location = saving_path + '/unwrapped/'+ str(coherence)
    xgrad_location = saving_path + '/x_grad/'  + str(coherence)
    ygrad_location = saving_path + '/y_grad/'  + str(coherence)
    noise_wrapped_location = saving_path + '/wrapped_noised/'  + str(coherence)
    unwrapped_location_noised = saving_path + '/unwrapped_noised/'  + str(coherence)
    xgrad_location_noised = saving_path + '/x_grad_noised/'  + str(coherence)
    ygrad_location_noised = saving_path + '/y_grad_noised/' + str(coherence)
    wrapped_location_filtered_saved = saving_path + 'wrapped_filtered/' + str(coherence)
    unwrapped_location_filtered = saving_path + '/unwrapped/'+ str(coherence)
    xgrad_location_filtered = saving_path + '/x_grad_filtered/'  + str(coherence)
    ygrad_location_filtered = saving_path + '/y_grad_filtered/'  + str(coherence)

    # import pdb; pdb.set_trace()
    phase_wrap_clean = phase_orig_s
    phase_unwrap_clean = phase_unwrap_s
    #noisy stuff
    phase_wrap_noise = phase_after_s
    
    k_values_noise = (phase_unwrap_clean - phase_wrap_noise)/(np.pi*2)
    k_values_noise = np.round_(k_values_noise)
    # k_values_noise = k_values_noise.filled(np.nan)
    shape = filtered_phase.shape[0]
    k_v_xn = expanded_array_func_x(k_values_noise, shape)
    k_v_yn = expanded_array_func_y(k_values_noise, shape)
    phase_grad_vertn = np.diff(k_v_yn, axis = 0) #get y direction gradients
    phase_grad_horn = np.diff(k_v_xn, axis = 1)  #get x direction gradients 
    
    
    k_values_clean = (phase_unwrap_clean - phase_wrap_clean)/(np.pi*2)
    k_values_clean = np.round_(k_values_clean)
   
    k_v_x = expanded_array_func_x(k_values_clean, shape)
    k_v_y = expanded_array_func_y(k_values_clean, shape)
    row, colomn = phase_wrap_clean.shape
    phase_grad_vert = np.diff(k_v_y, axis = 0) #get y direction gradients
    phase_grad_hor = np.diff(k_v_x, axis = 1)  #get x direction gradients 
    
    k_values_filtered = (phase_unwrap_clean - filtered_phase)/(np.pi*2)
    k_values_filtered = np.round_(k_values_filtered)
   
    k_v_xf = expanded_array_func_x(k_values_filtered, shape)
    k_v_yf = expanded_array_func_y(k_values_filtered, shape)
    row, colomn = filtered_phase.shape
    phase_grad_vertf = np.diff(k_v_yf, axis = 0) #get y direction gradients
    phase_grad_horf = np.diff(k_v_xf, axis = 1)  #get x direction gradients 
    
    each_file = str(count)
    save(wrapped_location_saved + '/' + each_file + '.npy', phase_wrap_clean)
    save(unwrapped_location + '/' + each_file+ '.npy', phase_unwrap_clean)
    save(xgrad_location + '/' + each_file  + '.npy', phase_grad_hor)
    save(ygrad_location + '/' + each_file + '.npy', phase_grad_vert)
    
    save(noise_wrapped_location + '/' + each_file + '.npy', phase_wrap_noise)
    save(xgrad_location_noised + '/' + each_file + '.npy', phase_grad_horn)
    save(ygrad_location_noised + '/'  + each_file + '.npy', phase_grad_vertn)

    save(wrapped_location_filtered_saved + '/' + each_file + '.npy', filtered_phase)
    save(xgrad_location_filtered + '/' + each_file + '.npy', phase_grad_horf)
    save(ygrad_location_filtered + '/' + each_file + '.npy', phase_grad_vertf)
   
    count +=1
    
    return count, phase_grad_hor, phase_grad_vert

#%%filtering steps
def magpha2RI_array(mag, pha):
    """Converts arrays of magnitude and phase to complex number array (real and imaginary)

    Args:
        mag (np.array): numpy array with magnitude values
        pha (np.array): numpy array with phase values

    Returns:
        np.array: complex number array
    """
    R = np.cos(pha) * mag
    I = np.sin(pha) * mag
    out = R + 1j*I
    return out

def pha2cpx(pha):
    """Creates normalised cpx interferogram from phase.
    (if xr.DataArray, it will again return xr.DataArray)
    """
    return np.exp(1j*pha)

def unit_circle(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return np.abs(dists<r).astype(int)
    
def nyquistmask(block):
    mask=np.zeros(block.shape) #should be square
    nyquistlen=int(mask.shape[0]/2+0.5) + 1 #+ extrapx
    circle=unit_circle(int(nyquistlen/2+0.5)) #will contain +1 px for zero
    i=int((mask.shape[0]-circle.shape[0])/2+0.5)
    j=int((mask.shape[1]-circle.shape[1])/2+0.5)
    mask[i:i+circle.shape[0],j:j+circle.shape[1]]=circle
    return mask

def coh_from_phadiff(phadiff, winsize=3):
    """Calculates coherence based on variance of interferogram, computed in window with given size

    Args:
        phadiff (np.ndarray): interferogram
        winsize (int): window size

    Returns:
        np.ndarray: coherence based on the variance

    """
    cpxdiff = pha2cpx(phadiff)
    variance = ndimage.generic_filter(np.real(cpxdiff), np.var, size=winsize)
    outcohr = 1 / np.sqrt(1 + 2*winsize * winsize * variance)
    variance = ndimage.generic_filter(np.imag(cpxdiff), np.var, size=winsize)
    outcohi = 1 / np.sqrt(1 + 2*winsize * winsize * variance)
    return (outcohr + outcohi) / 2

def wrap2phase(A):
    """Wraps array to -pi,pi (or 0,2pi?)
    """
    return np.angle(np.exp(1j*A))

def goldstein_filter_xr(inpha, blocklen=16, alpha=0.8, ovlpx=None, nproc=1, returncoh=True,
                        mask_nyquist=False):  # ovlwin=8, nproc=1):
    """Goldstein filtering of phase

    Args:
        inpha (xr.DataArray): array of phase (for now, the script will create cpx from phase)
        blocklen (int): size of rectangular window in pixels
        alpha (float): Goldstein alpha parameter - greater the alpha, greater the smoothing
        ovlpx (int): how many pixels should overlap the window
        nproc (int): number of processors to be used by dask
        returncoh (boolean): return coherence instead of the spectral magnitude

    Returns:
        xr.DataArray,xr.DataArray: filtered phase, magnitude (try np.log to use for masking)
    """
    # import pdb; pdb.set_trace()
    import dask.array as da
    
    if ovlpx == None:
        ovlpx = int(blocklen / 4)  # does it make sense? gamma recommends /8 but this might be too much?
    # dask works by adding extra pixels around the block window. thus calculate the central window here:
    blocklen = blocklen - ovlpx
    outpha = inpha.copy()
    if np.any(np.isnan(inpha)):
        inpha[np.isnan(inpha)] = 0
        incpx = pha2cpx(inpha)
    else:
        incpx = pha2cpx(inpha)
    winsize = (blocklen-ovlpx, blocklen-ovlpx)
    incpxb = da.from_array(incpx, chunks=winsize)
    # import pdb; pdb.set_trace()
    f = incpxb.map_overlap(goldstein_AHML, alpha=alpha, mask_nyquist=mask_nyquist, returnphadiff = returncoh,
                         depth=ovlpx, boundary='reflect',
                         meta=np.array((), dtype=np.complex128), chunks=(1, 1))
    cpxb = f.compute(num_workers=nproc)
    outpha = np.angle(cpxb)
    outmag = outpha.copy()
    outmag = np.abs(cpxb)
    outmag[outmag > 1] = 1 # just in case..
    if returncoh:
        # obsolete, will probably remove it
        print('better use specmag - we will probably remove the returncoh function')
        outmag = coh_from_phadiff(outmag-np.pi, 3)
    else:
        phadiff = outpha.copy()
        phadiff = wrap2phase(np.angle(incpx) - outpha)
        phadiff = coh_from_phadiff(phadiff)
        phadiff[np.isnan(phadiff)] = 0
        outmag = phadiff * outmag
    return outpha, outmag



def goldstein_AHML(block, alpha=0.8, kernelsigma=0.75, mask_nyquist=False, returnphadiff=True):
    cpx_fft = np.fft.fft2(block)
    # get 2d spectral magnitude of the block
    H = np.abs(cpx_fft)
    #firstfreq = H[0][0]   # useful to get avg coh if /block.shape
    H = np.fft.fftshift(H)
    # mask frequencies above Nyquist frequency
    if mask_nyquist:
        mask = nyquistmask(block)
        H = H*mask
    # phase ramps using masked H (i.e. low pass)
    # cpxm=np.fft.ifft2(cpx_fft*np.fft.fftshift(Hm))
    '''
    if returnphadiff: 
        # this is based on phase difference after convolution within Nyquist freq range - needs improvement, but it works
        phadiff = wrap2phase(np.angle(block) - np.angle(np.fft.ifft2(cpx_fft * np.fft.ifftshift(H))))  # C[0])
        #cc = 1 - coh_from_phadiff(phadiff, 3)
        #cpxfilt = magpha2RI_array(cc, np.angle(cpxfilt))
        return phadiff
    # perform cross-correlation of the original cpx block with the low-pass result
    cc = cpx_fft * np.conj(np.fft.fftshift(Hm))
    cc = cpx_fft * np.conj(np.fft.fftshift(H))
    cc = np.abs(np.fft.ifft2(cc))  # now i need to somehow normalise cc - not solved yet
    #
    # horrible solution, but maybe works?
    gh = wrap2phase(np.angle(block) - np.angle(np.fft.ifft2(cpx_fft * np.fft.ifftshift(Hm))))  # C[0])
    bgr = 1 - coh_from_phadiff(gh, 3)
    # avgcc=np.max(Hm)/32/32
    # avgcc=firstfreq/32/32
    # cc=cc/32/32
    # cc=10*np.log10(cc)*avgcc/32/32
    '''
    # only now convolve with Gaussian kernel to filter (not masking here, although we might consider it)
    kernel = Gaussian2DKernel(x_stddev=kernelsigma)  # sigma 1 gives 9x9 gaussian kernel
    #kernel = Gaussian2DKernel(x_stddev=kernelsigma, x_size = H.shape[1], y_size = H.shape[0] )
    #H = H * kernel.array
    H = convolve(H, kernel)   # but correctly i should only multiply with the gauss window, see above
    H = np.fft.ifftshift(H)
    # centering not needed? but maybe yes for mag/specmag
    #if meanH:
    meanH = np.median(H)
    if meanH != 0:
        H = H / meanH
    H = H ** alpha
    '''
    # try maxx it
    mask = nyquistmask(block)
    Hm = np.fft.ifftshift(H)*mask  # but here i am masking from centre, not from freq that appears most (max H)...
    noisesum = H.sum() - Hm.sum() + 0.001
    snr = Hm.sum()/noisesum
    #nsr = noisesum/Hm.sum()
    maxH = np.max(Hm)
    #maxH = np.max(H)
    #ratioH = block.shape[0]*block.shape[1]/maxH
    ratioH = 1/maxH
    #x = 1024/maxH  * valH
    H = H* ratioH *snr # / 1024)
    '''
    mask = nyquistmask(block)
    Hm = np.fft.ifftshift(H)*mask
    maxH = np.max(Hm)
    ratioH = 1/maxH
    Hr = H* ratioH # not bad try! but then some real dark areas as too bright then
    
    noisesum = H.sum() - Hm.sum() + 0.001
    #snr = Hm.sum()/noisesum
    nsr = 1-noisesum/H.sum()
    #Hs = H *snr
    Hn = H *nsr
    Hb = Hn * Hr #s * H
    H=Hr
    
    cpxfilt = np.fft.ifft2(cpx_fft * H)
    cpxfiltbad = np.fft.ifft2(cpx_fft * Hb)
    #cpxfilt = magpha2RI_array(np.abs(cpxfilt)*(1-nsr), np.angle(cpxfilt))
    cpxfilt = magpha2RI_array(np.abs(cpxfilt)*np.abs(cpxfiltbad), np.angle(cpxfilt))
    if returnphadiff:  # Oct 28, 2022: using the goldstein-filtered ck to get the phadiff (for coh measure, later)
        # this is based on phase difference after convolution within Nyquist freq range - needs improvement, but it works
        # recalc now, from the filtered version
        mask = nyquistmask(block)
        cpx_fft = np.fft.fft2(pha2cpx(np.angle(cpxfilt)))
        H = np.abs(cpx_fft)
        H = np.fft.fftshift(H)
        H = H * mask
        
        cpxnyquistfilt = np.fft.ifft2(cpx_fft * np.fft.ifftshift(H))
        phadiff = wrap2phase(np.angle(cpxfilt) - np.angle(cpxnyquistfilt))  # C[0])
        cpxfilt = magpha2RI_array(phadiff+np.pi, np.angle(cpxfilt))  # can mag be negative? i don't think so

    return cpxfilt

#%%running code

folder_list =  os.listdir(wrapped)
minimum = 0

coh_s = []
phase_obs_s = []
phase_orig_s = []
phase_unwrap = []
dem_s = []
phase_noise_s = []
phase_after_s = []
noise_std_s = []
cpx_orig_s = []
cpx_after_s = []
cpx_obs_s = []

wrapped_list_folder = os.listdir(wrapped)

for each_file in range(0, len(wrapped_list_folder)):
    # import pdb; pdb.set_trace()
    unwrapped_loaded = np.load(unwrapped + wrapped_list_folder[each_file])
    wrapped_loaded = np.load(wrapped + wrapped_list_folder[each_file])
    
    if len(np.unique(wrapped_loaded)) > 1:
        
        if 1 - (np.count_nonzero(np.isnan(wrapped_loaded)) / (112*112)) > 0.3:
    
            # import pdb; pdb.set_trace()
            coh = np.ones(wrapped_loaded.shape)
            threshold = random.uniform(0.3, 1.0)
            threshold = round(threshold, 2)
            coh = coh * threshold
        
            """ original phase image """
        
            phase_orig = wrapped_loaded
            phase_orig = np.copy(wrapped_loaded)
            cpx_orig = np.exp(1j*phase_orig)
            cpx_orig_s.append(cpx_orig)
            
            
            SNR = coh/(1-coh)
            std_dev = 1/np.sqrt(SNR)
            phase_noise = np.zeros(std_dev.shape)
            for index, value in np.ndenumerate(std_dev):
                elmt = np.random.normal(loc = 0.0, scale=value, size=1)
                phase_noise[index] = elmt[0]
        
            phase_after_noise_wrap = np.angle(np.exp(1j*phase_orig)*np.exp(1j*phase_noise))
            phase_noise_s = np.angle(np.exp(1j*phase_noise))
            
            if threshold <= 0.3:
                alpha = 3
            elif threshold <= 0.4:
                alpha = 2
            else:
                alpha = 0.8
            
            filtered_phase, otherthing = goldstein_filter_xr(phase_after_noise_wrap, alpha = alpha)
            phase_after_s = phase_after_noise_wrap
            # import pdb; pdb.set_trace()
            patch_size = (partition_length, partition_length)
            patches_wrapped = phase_orig #view_as_blocks(phase_orig, block_shape=patch_size).squeeze()
            patches_unwrapped = unwrapped_loaded# view_as_blocks(unwrapped_loaded, block_shape=patch_size).squeeze()
            patches_phase_noise_s =phase_noise_s# view_as_blocks(phase_noise_s, block_shape=patch_size).squeeze()
            patches_filtered_phase = filtered_phase#view_as_blocks(filtered_phase, block_shape=patch_size).squeeze()
            patches_phase_after_s = phase_after_s#view_as_blocks(phase_after_s, block_shape=patch_size).squeeze()
        
            # for i in range(patches_wrapped.shape[0]):
            #     for j in range(patches_wrapped.shape[1]):
                    
            patch_wrapped = patches_wrapped#[i, j]
            patch_unwrapped = patches_unwrapped#[i, j]
            patch_phase_noise_s = patches_phase_noise_s#[i, j]
            patch_filtered_phase = patches_filtered_phase#[i, j]
            patch_phase_after_s = patches_phase_after_s#[i, j]

            minimum, pgv, pgh = get_ambiguity_and_gradients(saving_path, patch_phase_after_s, patch_wrapped, patch_unwrapped, patch_phase_noise_s, minimum, patch_filtered_phase, threshold)  
            minimum += 1
            
            patch_size_total = phase_after_noise_wrap.shape[0] * phase_after_noise_wrap.shape[1]
            xgrad_extralocs = np.count_nonzero(pgh >= +2) + np.count_nonzero(pgh <= -2) 
            ygrad_extralocs = np.count_nonzero(pgv >= +2) + np.count_nonzero(pgv <= -2)
            
            per_xextremes = (xgrad_extralocs/patch_size_total)*100
            per_yextremes = (ygrad_extralocs/patch_size_total)*100
            comb_per = per_yextremes + per_xextremes
            #rotate 90oc
            
            if per_xextremes >= 2 or per_yextremes >= 2 or comb_per >= 3:
                minimum, pgv, pgh = get_ambiguity_and_gradients(saving_path, np.rot90(patch_phase_after_s), np.rot90(patch_wrapped), np.rot90(patch_unwrapped), np.rot90(patch_phase_noise_s), minimum, np.rot90(patch_filtered_phase), threshold)  
                minimum += 1
                
                # #rotate 90oc
                
                minimum, pgv, pgh = get_ambiguity_and_gradients(saving_path, np.rot90(np.rot90(patch_phase_after_s)), np.rot90(np.rot90(patch_wrapped)), np.rot90(np.rot90(patch_unwrapped)), np.rot90(np.rot90(patch_phase_noise_s)), minimum, np.rot90(np.rot90(patch_filtered_phase)), threshold)  
                minimum += 1
                
                # #rotate 90oc
                
                minimum, pgv, pgh = get_ambiguity_and_gradients(saving_path, np.rot90(np.rot90(np.rot90(patch_phase_after_s))), np.rot90(np.rot90(np.rot90(patch_wrapped))), np.rot90(np.rot90(np.rot90(patch_unwrapped))), np.rot90(np.rot90(np.rot90(patch_phase_noise_s))), minimum, np.rot90(np.rot90(np.rot90(patch_filtered_phase))), threshold)  
                minimum += 1
