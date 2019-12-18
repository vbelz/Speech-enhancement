# Speech-enhancement
---
[![Build Status](https://travis-ci.com/vbelz/Speech-enhancement.svg?branch=master)](https://travis-ci.com/vbelz/Speech-enhancement)
>
>
## Introduction

<img src="img/denoise_10classes.gif" alt="Spectrogram denoising" title="Speech enhancement"/>

This project aims at building a speech enhancement system to attenuate environmental noise.

Audios have many different ways to be represented, going from raw time series to time-frequency decompositions.
The choice of the representation is crucial for the performance of your system.
Among time-frequency decompositions, Spectrograms have been proved to be a useful representation for audio processing. They consist in 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame. In such they appear a natural domain to apply the CNNS architectures for images directly to sound. Between magnitude and phase spectrograms, magnitude spectrograms contain most the structure of the signal. Phase spectrograms appear to show only little temporal and spectral regularities.

In this project, I will use magnitude spectrograms as a representation of sound (cf image below) in order to predict the noise model to be subtracted to a noisy voice spectrogram.

<img src="img/sound_to_spectrogram.png" alt="sound representation" title="sound representation" />

The project is decomposed in three modes: `data creation`, `training` and `prediction`.

## Prepare the data

To create the datasets for training, I gathered english speech clean voices  and environmental noises from different sources.

The clean voices were mainly gathered from [LibriSpeech](http://www.openslr.org/12/): an ASR corpus based on public domain audio books. I used as well some datas from [SiSec](https://sisec.inria.fr/sisec-2015/2015-two-channel-mixtures-of-speech-and-real-world-background-noise/).
The environmental noises were gathered from [ESC-50 dataset](https://github.com/karoldvl/ESC-50) or [https://www.ee.columbia.edu/~dpwe/sounds/](https://www.ee.columbia.edu/~dpwe/sounds/).  

 For this project, I focused on 10 classes of environmental noise: tic clock, foot steps, bells, handsaw, alarm, fireworks, insects, brushing teeth, vaccum cleaner and snoring. These classes are illustrated in the image below
 (I created this image using pictures from (https://unsplash.com)[https://unsplash.com]).

<img src="img/classes_noise.png" alt="classes of environmental noise used" title="classes of environmental noise" />

To create the datasets for training/validation/testing, audios were sampled at 8kHz and I extracted windows
slighly above 1 second. I performed some data augmentation for the environmental noises (taking the windows at different times creates different noise windows). Noises have been blended to clean voices  with a randomization of the noise level (between 20% and 80%). At the end, training data consisted of 10h of noisy voice & clean voice,
and validation data of 1h of sound.

To prepare the data, I recommend to create data/Train and data/Test folders in a location separate from your code folder. Then create the following structure as in the image below:

<img src="img/structure_folder.png" alt="data folder structure" title="data folder structure" />

You would modify the `noise_dir`, `voice_dir`, `path_save_spectrogram`, `path_save_time_serie`, and `path_save_sound` paths name accordingly into the `args.py` file that takes the default parameters for the program.

Place your noise audio files into `noise_dir` directory and your clean voice files into `voice_dir`.

Specify how many frames you want to create as `nb_samples` in `args.py` (or pass it as argument from the terminal)
I let nb_samples=50 by default for the demo but for production I would recommend having 40 000 or more.

Then run `python main.py --mode='data_creation' ` . This will randomly blend some clean voices from `voice_dir` with some noises from `noise_dir` and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase, time series and sounds (for QC or to test other networks). It takes as inputs parameters defined in `args.py`. Parameters for STFT, frame length, hop_length can be modified in `args.py` (or pass it as argument from the terminal), but with the default parameters each window will be converted into spectrogram matrix of size 128 x 128.

Datasets to be used for training will be magnitude spectrograms of noisy voices and magnitude spectrograms of clean voices.


## Training

The model used for the training is a U-Net, a Deep Convolutional Autoencoder with symmetric skip connections. [U-Net](https://arxiv.org/abs/1505.04597) was initially developed for Bio Medical Image Segmentation. Here the U-Net has been adapted to denoise spectrograms.

As input to the network, the magnitude spectrograms of the noisy voices. As output the Noise to model (noisy voice magnitude spectrogram - clean voice magnitude spectrogram). Both input and output matrix are scaled with a global scaling to be mapped into a distribution between -1 and 1.

<img src="img/Unet_noisyvoice_to_noisemodel.png" alt="Unet training" title="Unet training" />

Many configurations have been tested during the training. For the preferred configuration the encoder is made of 10 convolutional layers (with LeakyReLU, maxpooling and dropout). The decoder is a symmetric expanding path. The last activation layer is a hyperbolic tangent (tanh) to have an output distribution between -1 and 1. For training from scratch the initial random weights where set with He normal initializer.

Model is compiled with Adam optimizer and the loss function used is the Huber loss as a compromise between the L1 and L2 loss.

Training on a modern GPU takes a couple of hours.

If you have a GPU for deep learning computation in your local computer, you can train with:
`python main.py --mode="training"`. It takes as inputs parameters defined in `args.py`. By default it will train from scratch (you can change this by turning `training_from_scratch` to false). You can
start training from pre-trained weights specified in `weights_folder` and `name_model`. I let available `model_unet.h5` with weights from my training in `./weights`. The number of epochs and the batch size for training are specified by `epochs` and `batch_size`. Best weights are automatically saved during training as `model_best.h5`. You can call fit_generator to only load part of the data to disk at training time.

Personally, I used the free GPU available at Google colab for my training. I let a notebook example at
`./colab/Train_denoise.ipynb`. If you have a large available space on your drive, you can load all your training data to your drive and load part of it at training time with the fit_generator option of tensorflow.keras. Personally I had limited space available on my Google drive so I pre-prepared in advanced batches of 5Gb to be loaded to drive for training. Weights were regularly saved and reload for next training.

At the end, I obtained a training loss of 0.002129 and a validation loss of 0.002406. Below a loss graph made in one of the trainings.

<img src="img/loss_training.png" alt="loss training" title="loss training" />

## Prediction

<img src="img/flow_prediction.png" alt="flow prediction part 1" title="flow prediction part 1" />

<img src="img/flow_prediction_part2.png" alt="flow prediction part 2" title="flow prediction part 2" />

[Input example bells](https://vbelz.github.io/Speech-enhancement/demo_data/validation/noisy_voice_bells28.wav)

[Predicted output example bells](https://vbelz.github.io/Speech-enhancement/demo_data/validation/voice_pred_bells28.wav)

[True output example bells](https://vbelz.github.io/Speech-enhancement/demo_data/validation/voice_bells28.wav)

<img src="img/validation_spec_examples.png" alt="validation examples" title="Spectrogram validation examples" />

<img src="img/validation_timeserie_examples.png" alt="validation examples timeserie" title="Time serie validation examples" />

<img src="img/denoise_ts_10classes.gif" alt="Timeserie denoising" title="Speech enhancement"/>

## How to use?

```
Clone this repository
pip install -r requirements.txt

```
## References

>Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.**Singing Voice Separation with Deep U-Net Convolutional Networks.** *ISMIR* (2017).
>
>[https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf]

>Grais, Emad M. and Plumbley, Mark D., **Single Channel Audio Source Separation using Convolutional Denoising Autoencoders** (2017).
>
>[https://arxiv.org/abs/1703.08019]

>Ronneberger O., Fischer P., Brox T. (2015) **U-Net: Convolutional Networks for Biomedical Image Segmentation**. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham
>
>[https://arxiv.org/abs/1505.04597]

> K. J. Piczak. **ESC: Dataset for Environmental Sound Classification**. *Proceedings of the 23rd Annual ACM Conference on Multimedia*, Brisbane, Australia, 2015.
>
> [DOI: http://dx.doi.org/10.1145/2733373.2806390]

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
