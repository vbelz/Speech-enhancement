import argparse

parser = argparse.ArgumentParser(description='Speech enhancement,data creation, training and prediction')

#mode to run the program (options: data creation, training or prediction)
parser.add_argument('--mode',default='prediction', type=str, choices=['data_creation', 'training', 'prediction'])
#folders where to find noise audios and clean voice audio to prepare training dataset (mode data_creation)
parser.add_argument('--noise_dir', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/noise', type=str)

parser.add_argument('--voice_dir', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/clean_voice', type=str)
#folders where to save spectrograms, time series and sounds for training / QC
parser.add_argument('--path_save_spectrogram', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/spectrogram/', type=str)

parser.add_argument('--path_save_time_serie', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/time_serie/', type=str)

parser.add_argument('--path_save_sound', default='/Users/vincentbelz/Documents/Data/Speech_enhancement/Train/sound/', type=str)
#How much frame to create in data_creation mode
parser.add_argument('--nb_samples', default=50, type=int)
#Training from scratch or pre-trained weights
parser.add_argument('--training_from_scratch',default=True, type=bool)
#folder of saved weights
parser.add_argument('--weights_folder', default='./weights', type=str)
#Nb of epochs for training
parser.add_argument('--epochs', default=10, type=int)
#Batch size for training
parser.add_argument('--batch_size', default=20, type=int)
#Name of saved model to read
parser.add_argument('--name_model', default='model_unet', type=str)
#directory where read noisy sound to denoise (prediction mode)
parser.add_argument('--audio_dir_prediction', default='./demo_data/test', type=str)
#directory to save the denoise sound (prediction mode)
parser.add_argument('--dir_save_prediction', default='./demo_data/save_predictions/', type=str)
#Noisy sound file to denoise (prediction mode)
parser.add_argument('--audio_input_prediction', default=['noisy_voice_long_t2.wav'], type=list)
#File name of sound output of denoise prediction
parser.add_argument('--audio_output_prediction', default='denoise_t2.wav', type=str)
# Sample rate chosen to read audio
parser.add_argument('--sample_rate', default=8000, type=int)
# Minimum duration of audio files to consider
parser.add_argument('--min_duration', default=1.0, type=float)
# Training data will be frame of slightly above 1 second
parser.add_argument('--frame_length', default=8064, type=int)
# hop length for clean voice files separation (no overlap)
parser.add_argument('--hop_length_frame', default=8064, type=int)
# hop length for noise files to blend (noise is splitted into several windows)
parser.add_argument('--hop_length_frame_noise', default=5000, type=int)
# Choosing n_fft and hop_length_fft to have squared spectrograms
parser.add_argument('--n_fft', default=255, type=int)

parser.add_argument('--hop_length_fft', default=63, type=int)
