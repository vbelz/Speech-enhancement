# Speech-enhancement
---
[![Build Status](https://travis-ci.com/vbelz/Speech-enhancement.svg?branch=master)](https://travis-ci.com/vbelz/Speech-enhancement)
>
>
## Introduction

<img src="img/denoise_10classes.gif" alt="Spectrogram denoising" title="Speech enhancement"/>

This project aims at building a speech enhancement system to attenuate environmental noise.

Audio have many different ways to be represented, going from raw time series to time-frequency decompositions (cf image below).
The choice of the representation is crucial for the performance of your system.
Among time-frequency decompositions, Spectrograms have been proved to be a useful representation for audio processing. They consist in 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame. In such they appear a natural domain to apply the CNNS architectures for images directly to sound.

In this project, i will use Spectrograms representation of sound (cf image below) in order to predict the noise model
to be subtracted to a noisy voice spectrogram.

<img src="img/sound_to_spectrogram.png" alt="sound representation" title="sound representation" />

The project is decomposed in three modes: `data creation`, `training` and `prediction`.

## Prepare the data
<img src="img/structure_folder.png" alt="data folder structure" title="data folder structure" />


<img src="img/classes_noise.png" alt="classes of environmental noise used" title="classes of environmental noise" />

## Training


<img src="img/Unet_noisyvoice_to_noisemodel.png" alt="Unet training" title="Unet training" />

## Prediction

<img src="img/flow_prediction.png" alt="flow prediction part 1" title="flow prediction part 1" />

<img src="img/flow_prediction_part2.png" alt="flow prediction part 2" title="flow prediction part 2" />

[Input example bells](/demo_data/validation/noisy_voice_bells28.wav)

[Predicted output example bells](https://vbelz.github.io/Speech-enhancement/demo_data/validation/voice_pred_bells28.wav)

[True output example bells](/demo_data/validation/voice_bells28.wav)

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
