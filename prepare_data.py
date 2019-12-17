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

list_noise_files = os.listdir(noise_dir)
list_voice_files = os.listdir(voice_dir)

def remove_ds_store(lst):
    """remove mac specific file if present"""
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')

    return lst

list_noise_files = remove_ds_store(list_noise_files)
list_voice_files = remove_ds_store(list_voice_files)

print(list_noise_files)
print(list_voice_files)
nb_voice_files = len(list_voice_files)
nb_noise_files = len(list_noise_files)

# Minimum duration of audio files to consider
min_duration = 1.0

# Our training data will be frame of slightly above 1 second
frame_length = 8064

# hop length for clean voice files separation (no overlap)
hop_length_frame = 8064

# hop length for noise files (we split noise into several windows)
hop_length_frame_noise = 5000


# Extracting noise and voice from folder and convert to numpy
noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                             frame_length, hop_length_frame_noise, min_duration)

voice = audio_files_to_numpy(voice_dir, list_voice_files,
                             sample_rate, frame_length, hop_length_frame, min_duration)

print(noise.shape)
print(voice.shape)
# How much frame to create
nb_samples = 50
# Blend some clean voices with random selected noises (and a random level of noise)
prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
    voice, noise, nb_samples, frame_length)

print(prod_voice.shape)
print(prod_noise.shape)
print(prod_noisy_voice.shape)
# To save the long audio generated to disk to QC:
noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'noisy_voice_long.wav', noisy_voice_long[0, :], sample_rate)
voice_long = prod_voice.reshape(1, nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'voice_long.wav', voice_long[0, :], sample_rate)
noise_long = prod_noise.reshape(1, nb_samples * frame_length)
librosa.output.write_wav(
    path_save_sound + 'noise_long.wav', noise_long[0, :], sample_rate)

# Choosing n_fft and hop_length_fft to have squared spectrograms
n_fft = 255
hop_length_fft = 63

dim_square_spec = int(n_fft / 2) + 1
print(dim_square_spec)

# Create Amplitude and phase of the sounds
m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
    prod_voice, dim_square_spec, n_fft, hop_length_fft)
m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
    prod_noise, dim_square_spec, n_fft, hop_length_fft)
m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
    prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

print(m_amp_db_voice.shape, m_pha_voice.shape, m_amp_db_noise.shape,
      m_pha_noise.shape, m_amp_db_noisy_voice.shape, m_pha_noisy_voice.shape)

# Save to disk for Training / QC
np.save(path_save_time_serie + 'voice_timeserie', prod_voice)
np.save(path_save_time_serie + 'noise_timeserie', prod_noise)
np.save(path_save_time_serie + 'noisy_voice_timeserie', prod_noisy_voice)


np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)
np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)

np.save(path_save_spectrogram + 'voice_pha_db', m_pha_voice)
np.save(path_save_spectrogram + 'noise_pha_db', m_pha_noise)
np.save(path_save_spectrogram + 'noisy_voice_pha_db', m_pha_noisy_voice)
