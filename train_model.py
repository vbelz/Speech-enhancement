import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_unet import unet
from data_tools import scaled_in, scaled_ou

path_save_spectrogram = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/spectrogram/'
weights_path = './weights/'
training_from_scratch = True

#load noisy voice & clean voice spectrograms created by prepare_data.py
X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
X_ou = np.load(path_save_spectrogram +'voice_amp_db'+".npy")
#Model of noise to predict
X_ou = X_in - X_ou

#Check distribution
print(stats.describe(X_in.reshape(-1,1)))
print(stats.describe(X_ou.reshape(-1,1)))

#to scale between -1 and 1
X_in = scaled_in(X_in)
X_ou = scaled_ou(X_ou)

#Check shape of spectrograms
print(X_in.shape)
print(X_ou.shape)
#Check new distribution
print(stats.describe(X_in.reshape(-1,1)))
print(stats.describe(X_ou.reshape(-1,1)))


#Reshape for training
X_in = X_in[:,:,:]
X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
X_ou = X_ou[:,:,:]
X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)

X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

#If training from scratch
if training_from_scratch:

    generator_nn=unet()
#If training from pre-trained weights
else:

    generator_nn=unet(pretrained_weights = weights_path+'model_unet.h5')


#Save best models to disk during training
checkpoint = ModelCheckpoint(weights_path+'model_best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

generator_nn.summary()
#Training
history = generator_nn.fit(X_train, y_train, epochs=10, batch_size=20, shuffle=True, callbacks=[checkpoint], verbose=1, validation_data=(X_test, y_test))

#Plot training and validation loss (log scale)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.yscale('log')
plt.title('Training and validation loss')
plt.legend()
plt.show()
