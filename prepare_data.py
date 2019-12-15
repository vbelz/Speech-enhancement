import os
import librosa
from data_tools import audio_files_to_numpy
from data_tools import blend_noise_randomly, numpy_audio_to_matrix_spectrogram
import numpy as np

# Sample rate chosen to read audio
sample_rate = 8000

noise_dir = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/noise'
voice_dir = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/clean_voice'
path_save_time_serie = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/time_serie/'
path_save_sound = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/sound/'
path_save_spectrogram = '/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/spectrogram/'

list_Noise_files = os.listdir(noise_dir)
list_Voice_files = os.listdir(voice_dir)

def remove_DS_store(lst):
    """remove mac specific file if present"""
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')

    return lst

list_Noise_files = remove_DS_store(list_Noise_files)
list_Voice_files = remove_DS_store(list_Voice_files)

print(list_Noise_files)
print(list_Voice_files)
Nb_voice_files = len(list_Voice_files)
Nb_noise_files = len(list_Noise_files)

# Minimum duration of audio files to consider
min_duration = 1.0

# Our training data will be frame of slightly above 1 second
frame_length = 8064

# hop length for clean voice files separation (no overlap)
hop_length_frame = 8064

# hop length for noise files (we split noise into several windows)
hop_length_frame_noise = 5000


# Extracting noise and voice from folder and convert to numpy
noise = audio_files_to_numpy(noise_dir, list_Noise_files, sample_rate,
                             frame_length, hop_length_frame_noise, min_duration)

voice = audio_files_to_numpy(voice_dir, list_Voice_files,
                             sample_rate, frame_length, hop_length_frame, min_duration)

print(noise.shape)
print(voice.shape)
# How much frame to create
Nb_samples = 50
# Blend some clean voices with random selected noises (and a random level of noise)
prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
    voice, noise, Nb_samples, frame_length)

print(prod_voice.shape)
print(prod_noise.shape)
print(prod_noisy_voice.shape)
# To save the long audio generated to disk to QC:
noisy_voice_long = prod_noisy_voice.reshape(1, Nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'noisy_voice_long.wav', noisy_voice_long[0, :], sample_rate)
voice_long = prod_voice.reshape(1, Nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'voice_long.wav', voice_long[0, :], sample_rate)
noise_long = prod_noise.reshape(1, Nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'noise_long.wav', noise_long[0, :], sample_rate)

# Choosing n_fft and hop_length_fft to have squared spectrograms
n_fft = 255
hop_length_fft = 63

dim_square_spec = int(n_fft / 2) + 1
print(dim_square_spec)

# Create Amplitude and phase of the sounds
M_amp_db_voice,  M_pha_voice = numpy_audio_to_matrix_spectrogram(
    prod_voice, dim_square_spec, n_fft, hop_length_fft)
M_amp_db_noise,  M_pha_noise = numpy_audio_to_matrix_spectrogram(
    prod_noise, dim_square_spec, n_fft, hop_length_fft)
M_amp_db_noisy_voice,  M_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
    prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

print(M_amp_db_voice.shape, M_pha_voice.shape, M_amp_db_noise.shape,
      M_pha_noise.shape, M_amp_db_noisy_voice.shape, M_pha_noisy_voice.shape)

# Save to disk for Training / QC
np.save(path_save_time_serie + 'voice_timeserie', prod_voice)
np.save(path_save_time_serie + 'noise_timeserie', prod_noise)
np.save(path_save_time_serie + 'noisy_voice_timeserie', prod_noisy_voice)


np.save(path_save_spectrogram + 'voice_M_amp_db', M_amp_db_voice)
np.save(path_save_spectrogram + 'noise_M_amp_db', M_amp_db_noise)
np.save(path_save_spectrogram + 'noisy_voice_M_amp_db', M_amp_db_noisy_voice)

np.save(path_save_spectrogram + 'voice_M_pha_db', M_pha_voice)
np.save(path_save_spectrogram + 'noise_M_pha_db', M_pha_noise)
np.save(path_save_spectrogram + 'noisy_voice_M_pha_db', M_pha_noisy_voice)
