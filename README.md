# Deep-fUS-artifacts: A deep learning network for functional ultrasound imaging of the brain for the reconstruction of motion artifact-affected compound frames

## Description

This repository contains the software to convert motion artifact-affected compound frames into Power Doppler images based on sagittal recordings provided by Emilie Mace (Max Planck Institute for Biological Intelligence). 

Functional ultrasound (fUS) is a rapidly emerging modality that enables whole-brain imaging of neural activity in awake and mobile rodents. To achieve sufficient blood flow sensitivity in the brain microvasculature, fUS relies on long ultrasound data acquisitions at high frame rates, posing high demands on the sampling and processing hardware. We developed an end-to-end image reconstruction approach based on deep learning that improves the image quality of the resulting Power Doppler images. We distorted compound images obtained under ideal conditions to train the CNN on the characteristics of motion artifacts and still compare them to the (optimal) ground truth images. Therefore, we used a Markov model to mimic translations and rotations over time and blurred individual compounds at random amplitudes.

The code in this repository has been tested with Python 3.7.7 using TensorFlow Keras version 2.1.0.

## Datasets
Training, validation, and testing datasets are made of pairs of compound data frames (x) and respective power Doppler images (y). An image size of 96×96 pixels is assumed. All the scripts in this repository assume that the data are provided in .mat format.

We have not yet made training and validation sets publicly available, as we're using unpublished data. Please reach out if you're interested.

## Data augmentation
To train a new model with a sparse dataset, it might make sense (e.g. when using the distortion options) to augment the available data samples. To do so, one may use the Matlab script `data_augmentation.m` in the [data](data) folder.

## Running the neural network
`python predict.py` can be executed to reconstruct images based on a pre-trained model. Use the 'pretrained_models' folder containing a .h5 file and copy it into the 'deep-fus-artifacts' folder. Use one of these models to reconstruct the power Doppler images stored in [data/test](data/test). `python train.py` can be used to train the model based on the datasets in the folders [data/train](data/train) and [data/dev](data/dev).

Note that both `python predict.py` and `python train.py` assume that the training, validation, and test datasets are available in the folders [data/train](data/train), [data/dev](data/dev), and [data/test](data/test), respectively. Both scripts plot the images reconstructed by the network (predicted), the reference images (original) reconstructed by the conventional power Doppler processing with the full compound dataset, the absolute difference images, and the scatter plots of the pixel values in the predicted and original images. In addition, the script saves the .mat files and the quantitative metrics (SSIM, PSNR, and NMSE) for each image. The results are saved in the model folder.

