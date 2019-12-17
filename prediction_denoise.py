import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio


path_weights = './weights/'

# load json and create model
json_file = open(path_weights+'model_unet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_weights+'model_unet.h5')
print("Loaded model from disk")

audio_dir_prediction = './demo_data/test'
audio_input = ['noisy_voice_long_t1.wav']
#audio_input = ['test_clean.wav']

dir_save_prediction = './demo_data/save_predictions/'

# Sample rate chosen to read audio
sample_rate = 8000

# Minimum duration of audio files to consider
min_duration = 1.0

# Our training data will be frame of slightly above 1 second
frame_length = 8064

# hop length for clean voice files separation (no overlap)
hop_length_frame = 8064

# hop length for noise files (we split noise into several windows)
hop_length_frame_noise = 5000


# Extracting noise and voice from folder and convert to numpy
audio = audio_files_to_numpy(audio_dir_prediction, audio_input, sample_rate,
                             frame_length, hop_length_frame, min_duration)

# Choosing n_fft and hop_length_fft to have squared spectrograms
n_fft = 255
hop_length_fft = 63

dim_square_spec = int(n_fft / 2) + 1
print(dim_square_spec)

# Create Amplitude and phase of the sounds
m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
    audio, dim_square_spec, n_fft, hop_length_fft)

#global scaling to have distribution -1/1
X_in = scaled_in(m_amp_db_audio)
#Reshape for prediction
X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
#Prediction using loaded network
X_pred = loaded_model.predict(X_in)
#Rescale back the noise model
inv_sca_X_pred = inv_scaled_ou(X_pred)
#Remove noise model from noisy speech
X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
#Reconstruct audio from denoised spectrogram and phase
print(X_denoise.shape)
print(m_pha_audio.shape)
print(frame_length)
print(hop_length_fft)
audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
#Number of frames
nb_samples = audio_denoise_recons.shape[0]
#Save all frames in one file
denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
librosa.output.write_wav(
    dir_save_prediction + 'denoise_t1.wav', denoise_long[0, :], sample_rate)
