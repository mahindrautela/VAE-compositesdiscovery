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
from lib.PCA import PCAmodel 

# coefficient of determination (R^2) for variance
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

# root mean squared difference
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# mean absolute percentage difference
def mape(y_true, y_pred):
        return 100*K.abs((y_true - y_pred)/(y_true+K.epsilon()))
    
# mean absolute error
def mae(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred), axis=-1)
    
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

imagesA0_1 = ImportImgData.load_A0_images(dfn_1, pathA0_1, imsize=128)
imagesA0_1 = imagesA0_1.astype("float32") / 255
imagesS0_1 = ImportImgData.load_S0_images(dfn_1, pathS0_1, imsize=128)
imagesS0_1 = imagesS0_1.astype("float32") / 255
img_merged_1 = np.concatenate([imagesS0_1,imagesA0_1],axis=3)
print("Shape of the dataset - 1",img_merged_1.shape)


##############################################################################
############################ location of the dataset-2 #######################
##############################################################################
# import images
pathS0_2 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/S0"
pathA0_2 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/" \
         "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Newdataset_15May2022/GrpVelBW/uni/A0"

# import labels
LabelPath_2 = "D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/"\
            "DispersionCalculatorCode/PropagationAngleProfiler_3/StoreResults/Labels/GenMaterials/Labels.txt"
            
# scale the properties
df_2 = pd.read_csv(LabelPath_2, header=None)
rho_2 = df_2.iloc[:,0:1]/100     # in (g/cm3)/10
E1_2 = df_2.iloc[:,1:2]*1e-9     # in GPa
E2_2 = df_2.iloc[:,2:3]*1e-9     # in GPa
G12_2 = df_2.iloc[:,3:4]*1e-9    # in GPa
v12_2 = df_2.iloc[:,4:5]*100     # in %
v23_2 = df_2.iloc[:,5:6]*100     # in %
dfn_2 = pd.concat([rho_2,E1_2,E2_2,G12_2,v12_2,v23_2],axis=1)

# load the images of A0, S0
imagesA0_2 = ImportImgData.load_A0_images(dfn_2, pathA0_2, imsize=128)
imagesA0_2 = imagesA0_2.astype("float32") / 255
imagesS0_2 = ImportImgData.load_S0_images(dfn_2, pathS0_2, imsize=128)
imagesS0_2 = imagesS0_2.astype("float32") / 255
img_merged_2 = np.concatenate([imagesS0_2,imagesA0_2],axis=3)
print("Shape of the dataset -2 ",img_merged_2.shape)


#############################################################################
###################### Training the generator (VAE) #########################
#############################################################################

# random no. generator for random_state in train_test_split on dataset-2
randno = random.randint(0, 42)
trainX,testX,trainY,testY = train_test_split(img_merged_2, dfn_2, test_size=0.10, random_state = 42)

# train and test images
print("Size of the training images",trainX.shape)
print("Size of the testing images",testX.shape)


# Definition of the network
(encoder, decoder) = network(128, 128, 2, 5)
encoder.summary()
decoder.summary()

# Training
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics = ['mape','mae'])
H = vae.fit(img_merged_2, epochs = 1500, batch_size=32)

# Summary of the loss
import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
plt.plot(H.history['reconstruction_loss'],'-o')
plt.title('Loss Curve',fontsize=22)
plt.ylabel('Loss',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Number of epochs',fontsize=22)
plt.axis([200,1500,0,50])
#plt.savefig("Loss.png", dpi=150)
plt.show()


#############################################################################
########## Load the supervised trained model (property prediction) ##########
#############################################################################

pathsavedmodel = 'D:/OneDrive - Indian Institute of Science/PhD-MSR/material_characterization_forJournal/pythoncodes/weights'
loaded_model = tf.keras.models.load_model(pathsavedmodel, custom_objects={'r_square': r_square,'rmse': rmse})


#############################################################################
######################### Latent space analysis  ############################
#############################################################################

latent_train_mu_2,latent_train_sigma_2,latent_train_2 = vae.encoder.predict(img_merged_2)
latent_test_mu_1,latent_test_sigma_1,latent_test_1 = vae.encoder.predict(img_merged_1)

from scipy.io import savemat
mdic1 = {"a1": latent_test_1, "label": "experiment"}
mdic2 = {"a2": latent_train_2, "label": "experiment"}
savemat("latent_1.mat", mdic1)
savemat("latent_2.mat", mdic2)

# Lets analyse the latent space bounds
plt.figure(figsize=(20,10))
plt.subplot(5,1,1)
xaxis = np.arange(0,latent_train_2.shape[0])
plt.plot(xaxis,latent_train_2[:,0])
plt.yticks(fontsize=15)
plt.subplot(5,1,2)
plt.plot(xaxis,latent_train_2[:,1])
plt.yticks(fontsize=15)
plt.subplot(5,1,3)
plt.plot(xaxis,latent_train_2[:,2])
plt.yticks(fontsize=15)
plt.subplot(5,1,4)
plt.plot(xaxis,latent_train_2[:,3])
plt.yticks(fontsize=15)
plt.subplot(5,1,5)
plt.plot(xaxis,latent_train_2[:,4])
plt.yticks(fontsize=15)

#############################################################################
######################### PCA of latent space  ##############################
#############################################################################

# pca = PCAmodel()
# nd = 2;

# (Xtr_pca, evr_tr_pca, recon_tr_pca) = pca.pcabuild(latent_train_2, nd)
# (Xte_pca, evr_te_pca, recon_te_pca) = pca.pcabuild(latent_test_1, nd)

    
# plt.figure(figsize=(6, 6))
# plt.scatter(Xtr_pca[:, 0], Xtr_pca[:, 1], c='green', marker=".")
# plt.scatter(Xte_pca[:, 0], Xte_pca[:, 1], c='red', marker="^")
# plt.legend(['Train (dataset-2)', 'Test (dataset-1)'], loc='best',fontsize=18)
# plt.xlabel('z1', fontsize=20)
# plt.ylabel('z2', fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()

#############################################################################
######################### t-SNE of latent space  ##############################
#############################################################################

from sklearn.manifold import TSNE
size_train = latent_train_2.shape[0]
latent_combine = np.vstack((latent_train_2,latent_test_1))

tsne = TSNE(n_components=2, learning_rate='auto', init ='pca').fit_transform(latent_combine)
X_train_tsne = tsne[0:size_train,:]
X_test_tsne  = tsne[size_train:,:]

plt.figure(figsize=(6, 6))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=(0, 0.4470, 0.7410), marker=".")
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c='#ff7f0e', marker="^")
plt.legend(['Train (dataset-2)', 'Test (dataset-1)'], loc='best',fontsize=14)
plt.xlabel('z1', fontsize=20)
plt.ylabel('z2', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

mdic1 = {"a3": X_train_tsne, "label": "experiment"}
mdic2 = {"a4": X_test_tsne, "label": "experiment"}
savemat("tsne_latent_1.mat", mdic1)
savemat("tsne_latent_2.mat", mdic2)

#############################################################################
########## Reconstruction of test set and inverse property prediction #######
#############################################################################

# FUNCTION --> reconstruction error
def reconstruction_error(samples, recon_samples):
    errors = []
    for (image, recon) in zip(samples, recon_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

# FUNCTION --> reconstruction of the images
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

# other metrics
mean_mae_A0 = np.mean(mae(dataset_1_A0,recon_A0))
mean_mae_S0 = np.mean(mae(dataset_1_S0,recon_S0))

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

#############################################################################
########## Generator of new samples from latent space and decoding ##########
#############################################################################

#########################  Global sampling ##################################

y_pred_store = []
Zg_store = []
gen_g_A0_store = []
gen_g_S0_store = []

for i in range(0,20):
    # sample a 5d point from latent space
    Zg = [random.uniform(-2,2),random.uniform(-2,2),random.uniform(-2,2),\
          random.uniform(-2,2),random.uniform(-2,2)]
    print('Global Sample from latent space = ',Zg)
    
    gen_zg = vae.decoder.predict([Zg])
    gen_g_S0 = gen_zg[:,:,:,0]
    gen_g_S0 = np.squeeze(gen_g_S0, axis=0)
    gen_g_A0 = gen_zg[:,:,:,1]
    gen_g_A0 = np.squeeze(gen_g_A0, axis=0)
    
    # plot the generated samples
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.imshow(gen_g_S0,cmap="gray")
    plt.title(str(np.round(Zg,2)))
    plt.subplot(2,1,2)
    plt.imshow(gen_g_A0,cmap="gray")
    plt.title(str(np.round(Zg,2)))
    
    # material properties of the generated sample
    img_g_A0 = gen_g_A0[np.newaxis,:,:,np.newaxis]
    img_g_S0 = gen_g_S0[np.newaxis,:,:,np.newaxis]
    y_pred = loaded_model.predict([img_g_A0*255, img_g_S0*255])
    print(y_pred)
    
    y_pred_store.append(y_pred)
    Zg_store.append(Zg)
    
    gen_g_A0_list = gen_g_A0.tolist()
    gen_g_S0_list = gen_g_S0.tolist()
    gen_g_A0_store.append(gen_g_A0_list)
    gen_g_S0_store.append(gen_g_S0_list)

y_pred_store = np.array(y_pred_store)
Zg_store = np.array(Zg_store)
gen_g_A0_store = np.array(gen_g_A0_store)
gen_g_S0_store = np.array(gen_g_S0_store)

mdic1 = {"vae_gen_matprop": y_pred_store, "label": "experiment"}
mdic2 = {"vae_gen_zg": Zg_store, "label": "experiment"}
mdic3 = {"vae_gen_A0": gen_g_A0_store, "label": "experiment"}
mdic4 = {"vae_gen_S0": gen_g_S0_store, "label": "experiment"}
savemat("vae_gen_matprop.mat", mdic1)
savemat("vae_gen_zg.mat", mdic2)
savemat("vae_gen_A0.mat", mdic3)
savemat("vae_gen_S0.mat", mdic4)

#########################  Local sampling ##################################

# Based on latent space bounds Z1,Z2,Z3 are
Z1bound = [-2,2]
Z2bound = [-2,2]
Z3bound = [-2,2]
Z4bound = [-2,2]
Z5bound = [-2,2]

# uniform sampling along directions
Zz = np.linspace(-2,2,17)

z10000 = []
z01000 = []
z00100 = []
z00010 = []
z00001 = []

for i in Zz:
    z10000.append([i,0,0,0,0])
    z01000.append([0,i,0,0,0])
    z00100.append([0,0,i,0,0])
    z00010.append([0,0,0,i,0])
    z00001.append([0,0,0,0,i])
        
z_all = [z10000,z01000,z00100,z00010,z00001]

# passing the sample to the decoder
y_pred_d_store = []
gen_d_A0_store = []
gen_d_S0_store = []
n = 4                                                                                                                                                                         
for j in z_all[n]:
    Zd = j
    print('Sample from latent space = ',Zd)
    gen_zd = vae.decoder.predict([Zd])
    gen_d_S0 = gen_zd[:,:,:,0]
    gen_d_S0 = np.squeeze(gen_d_S0, axis=0)
    gen_d_A0 = gen_zd[:,:,:,1]
    gen_d_A0 = np.squeeze(gen_d_A0, axis=0)

    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.imshow(gen_d_S0,cmap="gray")
    plt.title(str(np.round(Zd,2)))
    plt.subplot(2,1,2)
    plt.imshow(gen_d_A0,cmap="gray")
    plt.title(str(np.round(Zd,2)))
    
    # material properties of the generated sample
    img_d_A0 = gen_d_A0[np.newaxis,:,:,np.newaxis]
    img_d_S0 = gen_d_S0[np.newaxis,:,:,np.newaxis]
    y_pred_d = loaded_model.predict([img_d_A0*255, img_d_S0*255])
    print(y_pred_d)
    
    y_pred_d_store.append(y_pred_d)
    
    gen_d_A0_list = gen_d_A0.tolist()
    gen_d_S0_list = gen_d_S0.tolist()
    gen_d_A0_store.append(gen_d_A0_list)
    gen_d_S0_store.append(gen_d_S0_list)
    
y_pred_d_store = np.array(y_pred_d_store)
gen_d_A0_store = np.array(gen_d_A0_store)
gen_d_S0_store = np.array(gen_d_S0_store)

mdic1 = {"vae_gen_d_A0_z_"+str(n+1): gen_d_A0_store, "label": "experiment"}
savemat("vae_gen_dir_A0_z_"+str(n+1)+".mat", mdic1)
mdic2 = {"vae_gen_d_S0_z_"+str(n+1): gen_d_S0_store, "label": "experiment"}
savemat("vae_gen_dir_S0_z_"+str(n+1)+".mat", mdic2)

scale = [100,1e9,1e9,1e9,0.01,0.01]
y_pred_d_scale = y_pred_d_store*[100,1e9,1e9,1e9,0.01,0.01]

plt.figure()
plt.subplot(3,2,1)
plt.plot(y_pred_d_scale[:,:,0],'-o')
plt.title("Density")
plt.subplot(3,2,2)
plt.plot(y_pred_d_scale[:,:,1],'-o')
plt.title("E1")
plt.subplot(3,2,3)
plt.plot(y_pred_d_scale[:,:,2],'-o')
plt.title("E2")
plt.subplot(3,2,4)
plt.plot(y_pred_d_scale[:,:,3],'-o')
plt.title("G12")
plt.subplot(3,2,5)
plt.plot(y_pred_d_scale[:,:,4],'-o')
plt.title("nu12")
plt.subplot(3,2,6)
plt.plot(y_pred_d_scale[:,:,5],'-o')
plt.title("nu23")
plt.show()

mdic = {"vae_dir_gen_matprop_z_"+str(n+1): y_pred_d_scale, "label": "experiment"}
savemat("vae_dir_gen_matprop_z_"+str(n+1)+".mat", mdic)

## semantic arrows
z_all_arr = np.array(z_all).reshape(85,5)
size_train = latent_combine.shape[0]
latent_new = np.vstack((latent_combine,z_all_arr))
tsne = TSNE(n_components=2, learning_rate='auto', init ='pca').fit_transform(latent_new)
X_train_test_tsne = tsne[0:size_train,:]
X_latent_unitDir  = tsne[size_train:,:]

plt.figure(figsize=(6, 6))
plt.scatter(X_train_test_tsne[:, 0], X_train_test_tsne[:, 1], c=(0, 0.4470, 0.7410), marker=".")
plt.scatter(X_latent_unitDir[0:17, 0], X_latent_unitDir[0:17, 1], c='#ff7f0e', marker=".")
plt.legend(['Train (dataset-2)', 'Test (dataset-1)'], loc='best',fontsize=14)
plt.xlabel('z1', fontsize=20)
plt.ylabel('z2', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

mdic1 = {"tnse": X_latent_unitDir, "label": "experiment"}
savemat("tnse_latent_unitDir.mat", mdic1)