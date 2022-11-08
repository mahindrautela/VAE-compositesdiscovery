import numpy as np
from numpy import savetxt
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from lib.importdataset import ImportImgData
from lib.networkVAE2 import Sampling
from lib.networkVAE2 import VAE 

# coefficient of determination (R^2) for variance
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

# root mean squared difference
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Build the autoencoder network
def network(L,B,ch,z,filters=(16, 32, 64, 128, 256)):
    latent_dim = z
    
    # Build the encoder
    encoder_inputs = layers.Input(shape=(L,B,ch))
    cx = encoder_inputs
    for f in filters:
        # apply a CONV => RELU => BN operation
        cx = layers.Conv2D(f, (3, 3), strides = 2, padding="same")(cx)
        cx = layers.LeakyReLU(alpha=0.2)(cx)   
    x = layers.Flatten()(cx)
    x = layers.Dense(64, activation="relu")(x)
    
    x = layers.Dropout(0.2)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Build the decoder
    volumeSize = K.int_shape(cx)
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(volumeSize[1:]))(latent_inputs)
    cx = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)   
    for f in filters[::-1]: # filters[::-1] = [256,128,64,32,16]
        # apply a CONV_TRANSPOSE => RELU => BN operation
        cx = layers.Conv2DTranspose(f, (3, 3), strides = 2, padding="same")(cx)
        cx = layers.LeakyReLU(alpha=0.2)(cx)
    
    # apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
    decoder_outputs = layers.Conv2DTranspose(ch, (3, 3), activation="sigmoid", padding="same")(cx)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    return (encoder, decoder)
            
##############################################################################
############################ location of the dataset-2 #######################
##############################################################################

pathS0 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/S0"
pathA0 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/A0"
# import labels
LabelPath = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/"\
            "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Labels/GenMaterials/Labels.txt"
            
            
df = pd.read_csv(LabelPath, header=None)
rho = df.iloc[:,0:1]/100     # in (g/cm3)/10
E1 = df.iloc[:,1:2]*1e-9     # in GPa
E2 = df.iloc[:,2:3]*1e-9     # in GPa
G12 = df.iloc[:,3:4]*1e-9    # in GPa
v12 = df.iloc[:,4:5]*100     # in %
v23 = df.iloc[:,5:6]*100     # in %
dfn = pd.concat([rho,E1,E2,G12,v12,v23],axis=1)

# load the images of A0, S0
imagesA0 = ImportImgData.load_A0_images(dfn, pathA0, imsize=128)
imagesA0 = imagesA0.astype("float32") / 255
imagesS0 = ImportImgData.load_S0_images(dfn, pathS0, imsize=128)
imagesS0 = imagesS0.astype("float32") / 255
img_merged = np.concatenate([imagesS0,imagesA0],axis=3)
print("Shape of the dataset",img_merged.shape)

##############################################################################
############################ location of the dataset-1 #######################
##############################################################################
pathS0_1 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/GroupVelPolarPlots/BWPolarRep/DCmaterials/uni/S0/"
pathA0_1 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/GroupVelPolarPlots/BWPolarRep/DCmaterials/uni/A0/"
LabelPath_1 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/"\
            "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Labels/MaterialsFromDC/Labels190.txt"
            
df_1 = pd.read_csv(LabelPath_1, header=None)
rho_1 = df_1.iloc[:,0:1]/100     # in (g/cm3)/10
E1_1 = df_1.iloc[:,1:2]*1e-9     # in GPa
E2_1 = df_1.iloc[:,2:3]*1e-9     # in GPa
G12_1 = df_1.iloc[:,3:4]*1e-9    # in GPa
v12_1 = df_1.iloc[:,4:5]*100     # in %
v23_1 = df_1.iloc[:,5:6]*100     # in %
dfn_1 = pd.concat([rho_1,E1_1,E2_1,G12_1,v12_1,v23_1],axis=1)

# load the images of A0, S0
imagesA0_1 = ImportImgData.load_A0_images(dfn_1, pathA0_1, imsize=128)
imagesA0_1 = imagesA0_1.astype("float32") / 255
imagesS0_1 = ImportImgData.load_S0_images(dfn_1, pathS0_1, imsize=128)
imagesS0_1 = imagesS0_1.astype("float32") / 255
img_merged_1 = np.concatenate([imagesS0_1,imagesA0_1],axis=3)
print("Shape of the dataset - 1",img_merged_1.shape)

# random no. generator for random_state in train_test_split on dataset-2
randno = random.randint(0, 42)
trainX,testX,trainY,testY = train_test_split(img_merged, dfn, test_size=0.15, random_state = 42)

# train and test images
print("Size of the training images",trainX.shape)
print("Size of the testing images",testX.shape)

#fulldata = np.concatenate([trainX, testX], axis=0)
#print(fulldata.shape)

# Definition of the network
(encoder, decoder) = network(128, 128, 2, 3)
encoder.summary()
decoder.summary()

# Training
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics = ['mape','mae'])
H = vae.fit(img_merged, epochs = 1200, batch_size=32)

# Summary of the loss
import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
plt.plot(H.history['reconstruction_loss'],'-o')
plt.title('Loss Curve',fontsize=22)
plt.ylabel('Loss',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Number of epochs',fontsize=22)
#plt.axis([100,250,2.5e-5,1e-4])
#plt.savefig("Loss.png", dpi=150)
plt.show()

#############################################################################
########## Load the supervised trained model (property prediction) ##########
#############################################################################

pathsavedmodel = 'D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/pythoncodes/weights'
loaded_model = tf.keras.models.load_model(pathsavedmodel, custom_objects={'r_square': r_square,'rmse': rmse})

############################### LATENT SPACE ##################################
latent_train_mu,latent_train_sigma,latent_train = vae.encoder.predict(img_merged)
#latent_test_mu,latent_test_sigma,latent_test = vae.encoder.predict(testX)
latent_test_mu_1,latent_test_sigma_1,latent_test_1 = vae.encoder.predict(img_merged_1)

# visualize the overall latent space
plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
plt.title('Latent space' ,fontsize=22)
ax.scatter3D(latent_train[:,0],latent_train[:,1],latent_train[:,2], c='green',marker=".")
#ax.scatter3D(latent_test[:,0],latent_test[:,1],latent_test[:,2], c='green',marker=".")
ax.scatter3D(latent_test_1[:,0],latent_test_1[:,1],latent_test_1[:,2], c='red',marker="s")
plt.legend(['Train', 'Test','Dataset-1'], loc='best',fontsize=18)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='z', labelsize=20)
#ax.set_xlabel('Z1',fontsize=22)
#ax.set_ylabel('Z2',fontsize=22)
#ax.set_zlabel('Z3',fontsize=22)
ax.set_xlim3d(-4,4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(-4,4)
ax.view_init(-150, 40) 
plt.savefig("overall_latentspace"+".png", dpi=600)

# reconstruction of the images
def reconstruction_error(samples, recon_samples):
    errors = []
    for (image, recon) in zip(samples, recon_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

def plot_reconstruction(sample,recon_sample,idx,error,mode):
    fig = plt.figure(figsize = (10, 7.2)) 
    fig.add_subplot(211)
    plt.title("Original Image",fontsize=12)
    plt.imshow(sample[idx,:,:],cmap="gray")
    plt.xticks([0,20,40,60,80,100,120],fontsize=12)
    plt.yticks([0,20,40,60,80,100,120],fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.add_subplot(212)
    plt.title("Reconstruction, MSE = "+str(error),fontsize=12)
    plt.imshow(recon_sample[idx,:,:],cmap="gray")
    plt.xticks([0,20,40,60,80,100,120],fontsize=12)
    plt.yticks([0,20,40,60,80,100,120],fontsize=12)
    plt.savefig("recon_vae_"+mode+"_"+str(idx)+".png", dpi=600)
    return plt.show()

######################### RECONSTRUCTION OF TEST SET #########################
decoded_img = vae.decoder.predict([latent_test_1])
recon_S0 = decoded_img[:,:,:,0]
recon_A0 = decoded_img[:,:,:,1]

dataset_1_A0 = np.squeeze(imagesA0_1, axis=3)
dataset_1_S0 = np.squeeze(imagesS0_1, axis=3)

idx = random.randint(0,dataset_1_A0.shape[0])
print("idx = ",idx)

mse_A0 = reconstruction_error(dataset_1_A0,recon_A0)
mse_S0 = reconstruction_error(dataset_1_S0,recon_S0)
mean_mse_A0 = np.mean(mse_A0)
mean_mse_S0 = np.mean(mse_S0)
plot_reconstruction(dataset_1_A0,recon_A0,idx,np.round(mse_A0[idx],6),"A0")
plot_reconstruction(dataset_1_S0,recon_S0,idx,np.round(mse_S0[idx],6),"S0")

# prediction from loaded model == original images dataset-1
y_pred = loaded_model.predict([imagesA0_1*255, imagesS0_1*255])
y_pred = np.array(y_pred)
y_true = dfn_1
diff = y_pred - y_true
percentDiff = (diff / y_true) * 100
absPercentDiff = np.abs(percentDiff)
absPercentDiff = round(absPercentDiff,2)
print(absPercentDiff)
#savetxt('invSL_orign_DC.csv', absPercentDiff, delimiter=',')

# prediction from loaded model == reconstructed images dataset-1
imgA0 = recon_S0[:,:,:,np.newaxis]
imgS0 = recon_A0[:,:,:,np.newaxis]
y_pred = loaded_model.predict([imgA0*255, imgS0*255])
y_pred = np.array(y_pred)
y_true = dfn_1
diff = y_pred - y_true
percentDiff = (diff / y_true) * 100
absPercentDiff = np.abs(percentDiff)
absPercentDiff = round(absPercentDiff,2)
print(absPercentDiff)
#savetxt('invSL_recon_DC.csv', absPercentDiff, delimiter=',')

######################### GENERATION OF NEW SAMPLES #########################
# Lets analyse the latent space bounds
plt.figure(figsize=(20,10))
xaxis = np.arange(0,latent_train.shape[0])
plt.plot(xaxis,latent_train[:,0],c = 'blue')
plt.figure(figsize=(20,10))
plt.plot(xaxis,latent_train[:,1],c='orange')
plt.figure(figsize=(20,10))
plt.plot(xaxis,latent_train[:,2],c='green')

# Based on above figure Z1,Z2,Z3 we can define the bounds
Z1bound = [-2,2]
Z2bound = [-2,2]
Z3bound = [-2,2]

# Directional monte carlo
Z_100 = [random.uniform(-2,2),0, 0]
Z_010 = [0, random.uniform(-2,2), 0]
Z_001 = [0, 0, random.uniform(-2,2)]
Z_110 = [random.uniform(-2,2), random.uniform(-2,2), 0]
Z_011 = [0, random.uniform(-2,2), random.uniform(-2,2)]
Z_101 = [random.uniform(-2,2), 0, random.uniform(-2,2)]

# selected samples (reconstructed images available)
z100 = [[-1.386,0,0],[-0.145,0,0],[0.375,0,0],[0.637,0,0],[1.059,0,0]]
z010 = [[0,-1.676,0],[0,-0.713,0],[0,0.886,0],[0,1.04,0],[0,1.458,0]]
z001 = [[0,0,-1.944],[0,0,-0.191],[0,0,0.549],[0,0,1.426],[0,0,1.962]]

# passing the sample to the decoder
Zs = z001[4]
print('Sample from latent space = ',Zs)
gen_zs = vae.decoder.predict([Zs])
gen_S0 = gen_zs[:,:,:,0]
gen_S0 = np.squeeze(gen_S0, axis=0)
gen_A0 = gen_zs[:,:,:,1]
gen_A0 = np.squeeze(gen_A0, axis=0)

# plot the generated samples
plt.figure()
plt.imshow(gen_S0,cmap="gray")
plt.title(str([np.round(Zs[0],3),np.round(Zs[1],3),np.round(Zs[2],3)]))
#plt.savefig("mcsample_S0_"+str([np.round(Zs[0],3),np.round(Zs[1],3),np.round(Zs[2],3)])+".png", dpi=600)

plt.figure()
plt.imshow(gen_A0,cmap="gray")
plt.title(str([np.round(Zs[0],3),np.round(Zs[1],3),np.round(Zs[2],3)]))
#plt.savefig("mcsample_A0_"+str([np.round(Zs[0],3),np.round(Zs[1],3),np.round(Zs[2],3)])+".png", dpi=600)

# PERFORM PREDICTION ON GENERATED SAMPLES
imgA0 = gen_S0[np.newaxis,:,:,np.newaxis]
imgS0 = gen_A0[np.newaxis,:,:,np.newaxis]
y_pred = loaded_model.predict([imgA0*255, imgS0*255])
y_pred = np.array(y_pred)
print(y_pred)