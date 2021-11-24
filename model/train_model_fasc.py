import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from tqdm import tqdm_notebook, tnrange
from skimage.io import imshow
from skimage.transform import resize
# from skimage.morphology import label
# from skimage.feature import structure_tensor
from sklearn.model_selection import train_test_split
# from PIL import Image, ImageDraw
# import cv2

import tensorflow as tf

from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from custom_functions_training import conv2d_block, get_unet, IoU


# list of names of all images in the given path
im_width = 512
im_height = 512
idsF = next(os.walk("fasc_images_S"))[2]
print("Total no. of fascicle images = ", len(idsF))
XF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)
yF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)



# tqdm is used to display the progress bar
for n, id_ in tqdm_notebook(enumerate(idsF), total=len(idsF)):
    # Load images
    imgF = load_img("fasc_images_S/"+id_, color_mode = 'grayscale')
    x_imgF = img_to_array(imgF)
    x_imgF = resize(x_imgF, (512, 512, 1), mode = 'constant', preserve_range = True)
    # Load masks
    maskF = img_to_array(load_img("fasc_masks_S/"+id_, color_mode = 'grayscale'))
    maskF = resize(maskF, (512, 512, 1), mode = 'constant', preserve_range = True)
    # Normalise and store images
    XF[n] = x_imgF/255.0
    yF[n] = maskF/255.0

# Split data into training and validation
# X_trainF, X_validF, y_trainF, y_validF = train_test_split(XF, yF, test_size=0.1, random_state=42)
X_trainF, X_validF, y_trainF, y_validF = train_test_split(XF, yF, test_size=0.1)

# Compile the model
input_imgF = Input((im_height, im_width, 1), name='img')
modelF = get_unet(input_imgF, n_filters=32, dropout=0.25, batchnorm=True)
modelF.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", IoU])

# Set some training parameters (e.g. the name you want to give to your trained model)
callbacksF = [
    EarlyStopping(patience=7, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=7, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-fascSnippets2-nc.h5', verbose=1, save_best_only=True, save_weights_only=False), # Name your model (the .h5 part)
    CSVLogger('fasc2_training_losses.csv', separator=',', append=False)
]

resultsF = modelF.fit(X_trainF, y_trainF, batch_size=2, epochs=50, callbacks=callbacksF,\
                    validation_data=(X_validF, y_validF))

# Visualise the results of training
# Variables stored in results.history: val_loss, val_acc, val_IoU, loss, acc, IoU, lr
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax[0].plot(resultsF.history["loss"], label="Training loss")
ax[0].plot(resultsF.history["val_loss"], label="Validation loss")
ax[0].set_title('Learning curve')
ax[0].plot( np.argmin(resultsF.history["val_loss"]), np.min(resultsF.history["val_loss"]), marker="x", color="r", label="best model")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log_loss")
ax[0].legend();

ax[1].plot(resultsF.history["val_IoU"], label="Training IoU")
ax[1].plot(resultsF.history["IoU"], label="Validation IoU")
ax[1].set_title("IoU curve")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IoU score")
ax[1].legend();

# Predict on training and validations sets
preds_trainF = modelF.predict(X_trainF, verbose=1)
preds_valF = modelF.predict(X_validF, verbose=1)

# Threshold predictions (only keep predictions with a minimum level of confidence)
preds_train_tF = (preds_trainF > 0.5).astype(np.uint8)
preds_val_tF = (preds_valF > 0.5).astype(np.uint8)