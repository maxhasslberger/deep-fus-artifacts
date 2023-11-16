"""
File:     deep-fus/src/utils.py
Author:   Tommaso Di Ianni (todiian@stanford.edu)

Copyright 2021 Tommaso Di Ianni

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Import packages
import numpy as np
import tensorflow as tf
import json
import scipy.io as sio
import os
import matplotlib.pyplot as plt

from scipy.stats import multinomial, norm
from typing import List


def open_dat_file(path):
    """
    This function is used to open a .dat file and return its content as a numpy array.

    Arguments:
    path -- path to the .dat file -> string

    Returns:
    data -- content of the .dat file -> numpy array
    """

    # Open .dat file
    with open(path, 'rb') as f:
        data = np.fromfile(f)

    imageSize = (145, 128, -1)
    data = data.reshape(imageSize)
    # Display image
    plt.imshow(data[:, :, 23], cmap='gray')

    return data


def displace_img(img, transl, rot):
    """
    This function is used to displace an image by a given amount of pixels in the x and y directions.
    The image is zero-padded to avoid artifacts.

    Arguments:
    img -- image to be displaced -> [96, 96] -> float
    transl -- translational displacement vector in pixels -> [x, y] -> int
    rot -- rotation angle (origin) in degrees -> int

    Returns:
    img_disp -- displaced image -> [96, 96] -> float
    """

    # Initialize output array
    img_disp = np.zeros(img.shape)

    # Displace image - first rotation, then translation

    return img_disp


def equilibrium_distribution(p_transition):
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq


def markov_sequence(p_init: np.array, p_transition: np.array, sequence_length: int) -> List[int]:
    """
    Generate a Markov sequence based on p_init and p_transition.
    """
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)

    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states


def ar_gaussian_heteroskedastic_emissions(states, k, sigmas, img_set):
    emissions = []
    prev_loc = [0.0, 0.0, 0.0]  # shift x, shift y, rotation

    img_set_disp = np.zeros(img_set.shape)

    for state in states:
        e = norm.rvs(loc=k * np.array(prev_loc), scale=sigmas[state])
        emissions.append(e)
        prev_loc = e

        # Displace images
        e = np.round(e)
        img_set_disp[:, :, state] = displace_img(img_set[:, :, state], e[:2], e[2])

    return emissions, img_set_disp


def exe_markov(img_set, n_img):
    # Define Markov chain - state 0: no motion, state 1: motion
    # p_init = np.array([0.5, 0.5])
    rest2rest = 0.98
    motion2motion = 0.8
    p_transition = np.array([[rest2rest, 1 - rest2rest], [1 - motion2motion, motion2motion]])

    # Generate Markov sequence
    states = markov_sequence(None, p_transition, n_img)

    # Define AR(1) process
    k = 1
    sigmas = [[0.4, 0.4, 0.2], [3, 3, 3.5]]  # [state, [x, y, rot]]

    # Generate AR(1) process
    emissions, img_set_disp = ar_gaussian_heteroskedastic_emissions(states, k, sigmas, img_set)  # [n_img, [x, y, rot]]

    # Plot Markov sequence and AR(1) process
    emissions = np.array(emissions).T

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel('State')
    plt.plot(states)
    plt.subplot(4, 1, 2)
    plt.ylabel('dx (pix)')
    plt.plot(emissions[0])
    plt.subplot(4, 1, 3)
    plt.ylabel('dy (pix)')
    plt.plot(emissions[1])
    plt.subplot(4, 1, 4)
    plt.ylabel('drot (deg)')
    plt.xlabel('# Image')
    plt.plot(emissions[2])
    plt.show()
    plt.pause(1)

    return img_set_disp


def load_dataset_add_motion(dataset, n_img, m):
    """
    This function is used to load the training, validation, and test datasets.

    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    m -- number of sets to load. Select m sets after random permutation

    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each
                    dataset
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96

    print('Loading ' + str(m) + ' ' + dataset + ' examples.')

    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix, n_img))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    for k in range(m):
        # Load dataset
        data_dir = '../data/' + dataset + '/fr' + str(k + 1) + '.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        set_x_tmp = mat_contents['x'][:, :, :n_img]
        set_x_mov = exe_markov(set_x_tmp, n_img)
        set_x[idx] = set_x_mov
        set_y[idx] = mat_contents['y']

    print('    Done loading ' + str(m) + ' ' + dataset + ' examples.')

    return set_x, set_y


def lin_transition(img_tot, n_pix, comp):  # column-wise receptive field of compound frames -> transition

    sz = np.ceil((1.0 - comp) * n_pix).astype(int)
    sel_idx = np.zeros((n_pix * sz, img_tot))
    all_idx = np.array(range(n_pix * n_pix)).reshape(n_pix, n_pix)

    fin_start = np.floor(n_pix - sz)  # final starting index
    scale_fac = fin_start / (img_tot - 1)  # scale img_tot range to fin_start (max starting index)

    for i in range(img_tot):
        arg = np.floor(i * scale_fac).astype(int)
        tmp_idx = all_idx[:, arg:arg + sz].reshape(-1)

        sel_idx[:, i] = tmp_idx

    return sel_idx.astype(int)


def random_selection(img_tot, n_pix, comp):  # random selection of pixels per compound frame in sequence

    frac_tot = (np.floor((1.0 - comp) * n_pix * n_pix)).astype(int)
    sel_idx = np.zeros((frac_tot, img_tot))
    tmp_idx = np.array(range(n_pix * n_pix))

    for i in range(img_tot):
        # Shuffle ids for current compound frame
        np.random.shuffle(tmp_idx)
        tmp_idx = tmp_idx.T
        np.random.shuffle(tmp_idx)

        sel_idx[:, i] = tmp_idx[:frac_tot]

    return sel_idx.astype(int)


def load_dataset_v4(dataset, comp, m, sel_idx=None, comp_method='linear'):
    """
    This function is used to load the training, validation, and test datasets. The utilized pixels keep their original
    positions. The rest is zero-padded.

    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    comp -- 0 <= compression factor <= 1
    m -- number of sets to load. Select m sets after random permutation
    sel_idx -- indices of selected pixels. If None, generate index list based on comp_method
    comp_method -- compression method

    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each
                    dataset
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    img_tot = 250

    print('Loading ' + str(m) + ' ' + dataset + ' examples.')

    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix, img_tot))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    # Create random ids for each compound frame
    if sel_idx is None:  # If no indices are provided, create random indices
        if comp_method == 'linear':
            sel_idx = lin_transition(img_tot, n_pix, comp)
        elif comp_method == 'random':
            sel_idx = random_selection(img_tot, n_pix, comp)

    for k in range(m):
        # Load dataset
        data_dir = '../data/' + dataset + '/fr' + str(k + 1) + '.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        # Pick selected indices for each compound frame
        tmp_x = mat_contents['x'].reshape((n_pix * n_pix, img_tot))
        reduced_x = np.zeros((n_pix * n_pix, img_tot))
        for i in range(img_tot):
            reduced_x[sel_idx[:, i], i] = tmp_x[sel_idx[:, i], i]

        # Reshape into image format
        set_x[idx] = reduced_x.reshape((n_pix, n_pix, img_tot))

        set_y[idx] = mat_contents['y']

    print('    Done loading ' + str(m) + ' ' + dataset + ' examples.')

    return set_x, set_y, sel_idx


def load_dataset_v3(dataset, comp, m, sel_idx=None, comp_method='random'):
    """
    This function is used to load the training, validation, and test datasets. The frames are zero padded to 96x96
    pixels.

    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    comp -- 0 <= compression factor <= 1
    m -- number of sets to load. Select m sets after random permutation
    comp_method -- compression method

    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each
                    dataset
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    img_tot = 250

    print('Loading ' + str(m) + ' ' + dataset + ' examples.')

    # Initialize output arrays
    div_fac = 16
    new_2d_dims = (np.ceil(n_pix * np.sqrt(1 - comp) / div_fac) * div_fac).astype(int)
    set_x = np.zeros((m, n_pix, n_pix, img_tot))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    # Create random ids for each compound frame
    frac_tot = (np.floor((1.0 - comp) * n_pix * n_pix)).astype(int)
    if sel_idx is None:  # If no indices are provided, create random indices
        sel_idx = random_selection(img_tot, n_pix, frac_tot)

    for k in range(m):
        # Load dataset
        data_dir = '../data/' + dataset + '/fr' + str(k + 1) + '.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        # Pick selected indices for each compound frame
        tmp_x = mat_contents['x'].reshape((n_pix * n_pix, img_tot))
        reduced_x = np.zeros((n_pix * n_pix, img_tot))
        for i in range(img_tot):
            reduced_x[:frac_tot, i] = tmp_x[sel_idx[:, i].astype(int), i]

        # Zero padding to fit dimensions
        reduced_x[frac_tot:, :] = np.zeros((n_pix * n_pix - frac_tot, img_tot))

        # Reshape into image format
        set_x[idx] = reduced_x.reshape((n_pix, n_pix, img_tot))

        set_y[idx] = mat_contents['y']

    print('    Done loading ' + str(m) + ' ' + dataset + ' examples.')

    return set_x, set_y, sel_idx


def load_dataset_reduce(dataset, comp, m, sel_idx=None, comp_method='random'):
    """
    This function is used to load the training, validation, and test datasets. The frames are reduced on size based on
    the compression factor.

    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    comp -- 0 <= compression factor <= 1
    m -- number of sets to load. Select m sets after random permutation
    comp_method -- compression method

    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each
                    dataset
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    img_tot = 250

    print('Loading ' + str(m) + ' ' + dataset + ' examples.')

    # Initialize output arrays
    div_fac = 16
    new_2d_dims = (np.ceil(n_pix * np.sqrt(1 - comp) / div_fac) * div_fac).astype(int)
    set_x = np.zeros((m, new_2d_dims, new_2d_dims, img_tot))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    # Create random ids for each compound frame
    frac_tot = (np.floor((1.0 - comp) * n_pix * n_pix)).astype(int)
    if sel_idx is None:  # If no indices are provided, create random indices
        sel_idx = random_selection(img_tot, n_pix, frac_tot)

    for k in range(m):
        # Load dataset
        data_dir = '../data/' + dataset + '/fr' + str(k + 1) + '.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        # Pick selected indices for each compound frame
        tmp_x = mat_contents['x'].reshape((n_pix * n_pix, img_tot))
        reduced_x = np.zeros((new_2d_dims * new_2d_dims, img_tot))
        for i in range(img_tot):
            reduced_x[:frac_tot, i] = tmp_x[sel_idx[:, i].astype(int), i]

        # Zero padding to fit dimensions
        reduced_x[frac_tot:, :] = np.zeros((new_2d_dims * new_2d_dims - frac_tot, img_tot))

        # Reshape into image format
        set_x[idx] = reduced_x.reshape((new_2d_dims, new_2d_dims, img_tot))

        set_y[idx] = mat_contents['y']

    print('    Done loading ' + str(m) + ' ' + dataset + ' examples.')

    return set_x, set_y, sel_idx


def load_dataset_original(dataset, n_img, m):
    """
    This function is used to load the training, validation, and test datasets.
    
    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    n_img -- number of compounded RF images 
    m -- number of sets to load. Select m sets after random permutation
    
    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each  
                    dataset    
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96

    print('Loading ' +str(m) +' ' +dataset +' examples.')

    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix, n_img))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    # Shuffle set list
    np.random.seed(1)
    np.random.shuffle(data_list)

    for k in range(m):
        # Load dataset
        data_dir = '../data/' +dataset +'/fr' +str(k+1) +'.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        set_x[idx] = mat_contents['x'][:,:,:n_img]
        set_y[idx] = mat_contents['y']

    print('    Done loading ' +str(m) +' ' +dataset +' examples.')

    return set_x, set_y


def plot_and_stats(Yhat, Y, model_dir):
    """
    This function is used for plotting the original and predicted frame, and their difference. 
    The function also calculates the following metrics:
    -- NMSE
    -- nRMSE
    -- SSIM
    -- PSNR
    
    Arguments:
    Yhat -- Predicted examples
    Y -- Original examples (ground truth)
    model_dir -- Path to the folder containing the file 'my_model.h5'
    
    Returns:
    --
                    
    """

    # Dynamic range [dB]
    dr = 40

    loc_dir = model_dir +'/plot_and_stats'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)

    nmse  = []
    nrmse = []
    ssim  = []
    psnr  = []

    # Create dict to store metrics
    metrics = {};

    for idx in range(np.minimum(Yhat.shape[0],50)):

        ###################
        # CALCULATE METRICS
        ###################

        # Prep for metric calc
        y_true = tf.convert_to_tensor(Y[idx])
        y_pred = tf.convert_to_tensor(Yhat[idx])

        y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
        y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)

        # NMSE
        nmse_tmp = tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true))/tf.keras.backend.mean(tf.keras.backend.square(y_true))
        nmse_tmp = np.float_(nmse_tmp)
        nmse.append(nmse_tmp)

        # nRMSE
        nrmse_tmp = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true)))/(tf.keras.backend.max(y_true)-tf.keras.backend.min(y_true))
        nrmse.append(np.float_(nrmse_tmp))

        # SSIM
        ssim_tmp = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
        ssim_tmp = np.float_(ssim_tmp)
        ssim.append(ssim_tmp)

        # Prep for PSNR calc
        y_pred = tf.divide(y_pred,tf.reduce_max(y_true))    # Normalize y_pred [0 1]
        y_pred = tf.clip_by_value(y_pred, np.power(10,-dr/10), 1)         # Clip to dynamic range
        y_pred = tf.multiply(tf.divide(tf.math.log(y_pred), tf.math.log(tf.constant(10, dtype=y_true.dtype))), 10)
        y_pred = (y_pred+dr)/dr

        y_true = tf.divide(y_true,tf.reduce_max(y_true))    # Normalize y_true [0 1]
        y_true = tf.clip_by_value(y_true, np.power(10,-dr/10), 1)          # Clip to dynamic range
        y_true = tf.multiply(tf.divide(tf.math.log(y_true), tf.math.log(tf.constant(10, dtype=y_true.dtype))), 10)
        y_true = (y_true+dr)/dr

        # PSNR
        psnr_tmp = tf.image.psnr(y_true, y_pred, max_val=1)
        psnr_tmp = np.float_(psnr_tmp)
        psnr.append(psnr_tmp)

        ###########################
        # PLOT ORIG AND PRED FRAMES
        ###########################

        # Convert Y to dB scale
        Y_dB = 10*np.log10(Y[idx]/np.amax(Y[idx]))

        # Clip to dynamic range
        Y_dB[np.where(Y_dB<=-dr)] = -dr
        Y_dB[np.isnan(Y_dB)] = -dr

        # Convert Yhat to dB scale
        Yhat_dB = 10*np.log10(Yhat[idx]/np.amax(Y[idx]))

        # Clip to dynamic range
        Yhat_dB[np.where(Yhat_dB<=-dr)] = -dr
        Yhat_dB[np.isnan(Yhat_dB)] = -dr

        # PLot Y
        fig, ax = plt.subplots()
        cs = ax.imshow(Y_dB, vmin=-dr, vmax=0, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Original ' +str(idx))
        plt.savefig(loc_dir +'/orig' +str(idx) +'.png')
        plt.close(fig)

        # Plot Yhat
        fig, ax = plt.subplots()
        cs = ax.imshow(Yhat_dB, vmin=-dr, vmax=0, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Pred ' +str(idx) +' - SSIM: ' +'{:.03f}'.format(ssim_tmp) +' - PSNR: ' +'{:.03f}'.format(psnr_tmp) +' - NMSE: ' +'{:.03f}'.format(nmse_tmp) +' - NRMSE: ' +'{:.03f}'.format(nrmse_tmp) )
        plt.savefig(loc_dir +'/pred' +str(idx) +'.png')
        plt.close(fig)

        # Plot difference
        img_diff = np.abs(Yhat_dB-Y_dB)
        fig, ax = plt.subplots()
        cs = ax.imshow(img_diff, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Difference ' +str(idx))
        plt.savefig(loc_dir +'/diff' +str(idx) +'.png')
        plt.close(fig)

        # Scatter plot
        y1 = np.copy(Y_dB)
        y2 = np.copy(Yhat_dB)
        fig, ax = plt.subplots()
        plt.scatter(y1.flatten(), y2.flatten(), marker='o', color='black')
        x = np.linspace(-40, 0, 41)
        plt.plot(x, x);
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.show()
        plt.savefig(loc_dir +'/scatt' +str(idx) +'.png')
        plt.close(fig)

    ######################
    # SAVE METRICS TO FILE
    ######################

    metrics["nmse"] = list(np.float_(nmse))
    metrics["nmse_mean"] = np.float_(np.mean(nmse))
    metrics["nmse_std"] = np.float_(np.std(nmse))

    metrics["nrmse"] = list(np.float_(nrmse))
    metrics["nrmse_mean"] = np.float_(np.mean(nrmse))
    metrics["nrmse_std"] = np.float_(np.std(nrmse))

    metrics["ssim"] = list(np.float_(ssim))
    metrics["ssim_mean"] = np.float_(np.mean(ssim))
    metrics["ssim_std"] = np.float_(np.std(ssim))

    metrics["psnr"] = list(np.float_(psnr))
    metrics["psnr_mean"] = np.float_(np.mean(psnr))
    metrics["psnr_std"] = np.float_(np.std(psnr))

    with open(loc_dir +'/metrics', 'w') as file:
        json.dump(metrics, file)

    return


def load_dataset_postproc(dataset, n_img, m):
    """
    This function is used to load the training, validation, and test datasets for the experiment
    using pre-processed power Doppler images.
    
    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    n_img -- number of compounded RF images 
    m -- number of sets to load. Select m sets after random permutation
    
    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each  
                    dataset    
    """

    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96

    print('Loading ' +str(m) +' ' +dataset +' examples.')

    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix))
    set_y = np.zeros((m, n_pix, n_pix))

    data_list = [i for i in range(m)]

    # Shuffle set list
    np.random.seed(1)
    np.random.shuffle(data_list)

    for k in range(m):
        # Load dataset
        data_dir = '../data/' +dataset +'_process/' +str(n_img) +'img/fr' +str(k+1) +'.mat'
        mat_contents = sio.loadmat(data_dir)

        idx = data_list[k]

        set_x[idx] = mat_contents['x']
        set_y[idx] = mat_contents['y']

    print('    Done loading ' +str(m) +' ' +dataset +' examples.')

    return set_x, set_y
