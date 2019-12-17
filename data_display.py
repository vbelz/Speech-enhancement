import matplotlib.pyplot as plt
import librosa.display

def make_plot_spectrogram(stftaudio_magnitude_db,sample_rate, hop_length_fft) :
    """This function plots a spectrogram"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(stftaudio_magnitude_db, x_axis='time', y_axis='linear',
                             sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    title = 'hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'
    plt.title(title.format(hop_length_fft,
                           stftaudio_magnitude_db.shape[1],
                           stftaudio_magnitude_db.shape[0],
                           stftaudio_magnitude_db.shape));
    return

def make_plot_phase(stft_phase,sample_rate,hop_length_fft) :
    """This function plots the phase in radian"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(np.angle(stft_phase), x_axis='time', y_axis='linear',
                             sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    title = 'hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'
    plt.title(title.format(hop_length_fft,
                           stft_phase.shape[1],
                           stft_phase.shape[0],
                           stft_phase.shape));
    return

def make_plot_time_serie(audio,sample_rate):
    """This function plots the audio as a time serie"""
    plt.figure(figsize=(12, 6))
    #plt.ylim(-0.05, 0.05)
    plt.title('Audio')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    librosa.display.waveplot(audio, sr=sample_rate)
    return


def make_3plots_spec_voice_noise(stftvoicenoise_mag_db,stftnoise_mag_db,stftvoice_mag_db,sample_rate, hop_length_fft):
    """This function plots the spectrograms of noisy voice, noise and voice as a single plot """
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.title('Spectrogram voice + noise')
    librosa.display.specshow(stftvoicenoise_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Spectrogram predicted voice')
    librosa.display.specshow(stftnoise_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Spectrogram true voice')
    librosa.display.specshow(stftvoice_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.tight_layout()

    return


def make_3plots_phase_voice_noise(stftvoicenoise_phase,stftnoise_phase,stftvoice_phase,sample_rate, hop_length_fft):
    """This function plots the phase in radians of noisy voice, noise and voice as a single plot """
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.title('Phase (radian) voice + noise')
    librosa.display.specshow(np.angle(stftvoicenoise_phase), x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Phase (radian) predicted voice')
    librosa.display.specshow(np.angle(stftnoise_phase), x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Phase (radian) true voice')
    librosa.display.specshow(np.angle(stftvoice_phase), x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.tight_layout()

    return


def make_3plots_timeseries_voice_noise(clipvoicenoise,clipnoise,clipvoice, sample_rate) :
    """This function plots the time series of audio of noisy voice, noise and voice as a single plot """
    #y_ax_min = min(clipvoicenoise) - 0.15
    #y_ax_max = max(clipvoicenoise) + 0.15

    plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=0.35)
    plt.subplot(3, 1, 1)
    plt.title('Audio voice + noise')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    librosa.display.waveplot(clipvoicenoise, sr=sample_rate)
    plt.ylim(-0.05, 0.05)
    plt.subplot(3, 1, 2)
    plt.title('Audio predicted voice')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    librosa.display.waveplot(clipnoise, sr=sample_rate)
    plt.ylim(-0.05, 0.05)
    plt.subplot(3, 1, 3)
    plt.title('Audio true voice')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    librosa.display.waveplot(clipvoice, sr=sample_rate)
    plt.ylim(-0.05, 0.05)

    return
