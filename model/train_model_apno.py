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

# Images will be re-scaled
im_width = 512
im_height = 512
border = 5

if __name__ == '__main__':

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # list of all images in the path
    ids = next(os.walk("apo_images"))[2]
    print("Total no. of aponeurosis images = ", len(ids))
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    # tqdm is used to display the progress bar
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img("apo_images/" + id_, color_mode='grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (512, 512, 1), mode='constant', preserve_range=True)
        # Load masks
        mask = img_to_array(load_img("apo_masks/" + id_, color_mode='grayscale'))
        mask = resize(mask, (512, 512, 1), mode='constant', preserve_range=True)
        # Normalise and store images
        X[n] = x_img / 255.0
        y[n] = mask / 255.0

    # Split data into training and validation
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)  # i.e. 90% training / 10% test split

    # Compile the aponeurosis model
    input_img = Input((im_height, im_width, 1), name='img')
    model_apo = get_unet(input_img, n_filters=64, dropout=0.25, batchnorm=True)
    model_apo.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", IoU])

    # Show a summary of the model structure
    model_apo.summary()

    # Set some training parameters
    callbacks = [
        EarlyStopping(patience=8, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-apo2-nc.h5', verbose=1, save_best_only=True, save_weights_only=False),
        # Give the model a name (the .h5 part)
        CSVLogger('apo2_weights.csv', separator=',', append=False)
    ]

    results = model_apo.fit(X_train, y_train, batch_size=2, epochs=60, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))

    # Variables stored in results.history: val_loss, val_acc, val_IoU, loss, acc, IoU, lr
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(results.history["loss"], label="Training loss")
    ax[0].plot(results.history["val_loss"], label="Validation loss")
    ax[0].set_title('Learning curve')
    ax[0].plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
               label="best model")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log_loss")
    ax[0].legend();

    ax[1].plot(results.history["val_IoU"], label="Training IoU")
    ax[1].plot(results.history["IoU"], label="Validation IoU")
    ax[1].set_title("IoU curve")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("IoU score")
    ax[1].legend();

    # Predict on training and validations sets
    preds_train = model_apo.predict(X_train, verbose=1)
    preds_val = model_apo.predict(X_valid, verbose=1)

    # Threshold predictions (only keep predictions with a minimum level of confidence)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)