import tensorflow as tf
print(tf.__version__)

# import the necessary packages
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from lib.importdataset import ImportImgData
from lib.networkVAE import CVAE

# loss functions
def L_t(y_true, y_pred):
    recon_1 = K.sum(K.binary_crossentropy(y_true, y_pred), axis = -1)
    recon_2 = K.mean(K.square((y_true - y_pred)))
    kl_loss = -0.5 * (1 + sigma - K.square(mu) - K.exp(sigma))
    kl_loss = K.mean(K.sum(kl_loss, axis = 1))
    return recon_2 + kl_loss

def L_kl(y_true, y_pred):
    kl_loss = -0.5 * (1 + sigma - K.square(mu) - K.exp(sigma))
    kl_loss = K.mean(K.sum(kl_loss, axis = 1))
    return kl_loss

def L_bce(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis = -1)

def L_mse(y_true, y_pred):
    return K.mean(K.square((y_true - y_pred)))

# callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('val_loss') < 1e-6) and (logs.get('loss') < 1e-6):
        print("\nReached perfect loss so cancelling training!")
        self.model.stop_training = True

epoch_schedule = myCallback()

# Metrics
# coefficient of determination (R^2) for variance
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

# root mean squared difference
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    

# location of the dataset
pathS0 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/S0"
pathA0 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/A0"
         
# import labels
LabelPath = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/"\
            "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Labels/GenMaterials/Labels.txt"

df = pd.read_csv(LabelPath, header=None)
print(df.shape)

rho = df.iloc[:,0:1]/100     # in (g/cm3)/10
E1 = df.iloc[:,1:2]*1e-9     # in GPa
E2 = df.iloc[:,2:3]*1e-9     # in GPa
G12 = df.iloc[:,3:4]*1e-9    # in GPa
v12 = df.iloc[:,4:5]*100     # in %
v23 = df.iloc[:,5:6]*100     # in %

dfn = pd.concat([rho,E1,E2,G12,v12,v23],axis=1)

# load the images of A0, S0
imagesA0 = ImportImgData.load_A0_images(dfn, pathA0, imsize=128)
imagesS0 = ImportImgData.load_S0_images(dfn, pathS0, imsize=128)
img_merged = np.concatenate([imagesS0,imagesA0],axis=3)
print("Shape of the dataset",img_merged.shape)

# random no. generator for random_state in train_test_split
from sklearn.model_selection import train_test_split
import random
randno = random.randint(0, 42)
trainX, testX = train_test_split(img_merged, test_size=0.15, random_state=randno)

# train and test images
print("Size of the training images",trainX.shape)
print("Size of the testing images",testX.shape)

# network architecture
(encoder, decoder, vae, mu, sigma) = CVAE.build(128, 128, 2)
encoder.summary()
decoder.summary()

# Training
from tensorflow.keras.optimizers import Adam

EPOCHS = 500
INIT_LR = 1e-3
BS = 64

opt = Adam(learning_rate = INIT_LR)
vae.compile(optimizer = opt, loss = L_t, metrics = [L_mse, L_kl])

H = vae.fit(trainX, trainX, 
            validation_data = (testX, testX), 
            epochs = EPOCHS, 
            batch_size = BS,
            verbose = 1)