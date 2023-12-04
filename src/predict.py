
# Import packages
import tensorflow as tf
import json
import os
import scipy.io as sio
from utils import *
from losses import *

###################
# PRE-TRAINED MODEL
###################

# model_dir = '../pretrained_models/ResUNet53D_125'
model_dir = '../../pretrained_models/custom'
n_img = 100

#####################
# LOAD MODEL AND DATA
#####################

m = 3

# Load model 
model = tf.keras.models.load_model(model_dir +'/my_model.h5', custom_objects={'loss': custom_loss(beta=0.1), 'ssim': ssim, 'psnr': psnr, 'nmse': nmse, 'nrmse': nrmse})
    
# Load TEST examples
X_test, Y_test = load_dataset_add_motion('test', n_img, m, distort=False)

# Standardize X data - use mean and standard deviation of training set
Xmean = -0.5237595494149918
Xstd = 131526.6016974602

X_test = (X_test-Xmean) / Xstd

##################
# PREDICT AND PLOT
##################

# Predict TEST examples
Yhat_test = model.predict(X_test, verbose=0)
    
# Plot original and predicted TEST examples
plot_and_stats(Yhat_test, Y_test, model_dir)

#################
# SAVE .MAT FILES
#################

mat_dir = model_dir +'/mat_files'
if not os.path.exists(mat_dir):
    os.mkdir(mat_dir)

fr_data = {}
for idx in range(m):
    fr_data['y_true'] = Y_test[idx]
    fr_data['y_pred'] = Yhat_test[idx]
    
    # Save dataset
    fr_str = 'fr' +str(idx) +'.mat'
    data_dir = os.path.join(os.getcwd(), mat_dir, fr_str)
    sio.savemat(data_dir, fr_data)