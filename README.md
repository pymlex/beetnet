# BeetNet — GRU Autoencoder for Mel Spectrogram Generation from Beethoven Piano Music

<img width="1106" height="350" alt="Screenshot 2025-12-31 230943" src="https://github.com/user-attachments/assets/01cc94b1-1cdf-481a-bc91-1e8de2fe08de" />

## Introduction

This project extracts two second audio segments from a curated [subset](https://www.kaggle.com/datasets/alexzyukov/beethoven) of [MusicNet](https://www.kaggle.com/datasets/alonhaviv/musicnet) containing solo piano works by Beethoven, converts audio to mel spectrograms, and trains a recurrent autoencoder to reconstruct spectrograms frame by frame. The model treats each spectrogram as a temporal sequence of frequency frames, encodes the sequence into a fixed size latent vector, and decodes the latent vector back into a full spectrogram. Reconstruction quality is measured with L1 loss in decibels and checkpoints are saved after each epoch.

## Data and preprocessing

The dataset contains 18941 audio segments, all mono, sampled at 16000 samples per second, each segment length equal to 2 seconds. Audio is resampled to 16000 Hz and normalized by peak amplitude before spectrogram computation.

Mel spectrogram parameters are:

- sample rate equal to 16000,
- FFT size equal to 2048,
- hop length equal to 512,  
- number of mel bins equal to 96.

Spectrogram amplitudes are converted to decibels, clipped to a dynamic range defined by $m_{\min}$ and $m_{\max}$, and standardised using global mean $\mu$ and standard deviation $\sigma$ computed across the whole dataset. Then the standartisations is applied. Clipped and standardised spectrogram tensors are cached in the directory `mel_cache` in PyTorch tensor format for fast data loading.

---

## Model architecture

The autoencoder consists of an encoder and a decoder built with GRU layers.

### Encoder 

An input mel spectrogram is interpreted as a sequence of time frames, each frame represented by a vector of length equal to the number of mel bins. Each frame is projected by a learnable linear layer into an encoding dimension and processed by a bidirectional stack of GRU layers. The final hidden state at the last time step is used as latent vector $z$ of dimension 256.

### Decoder

The decoder receives the latent vector $z$, projects $z$ into a decoder hidden space, repeats the projected vector for every time step of the target sequence, feeds the repeated sequence into a GRU stack and applies a linear output layer at each time step to predict mel coefficients. The decoder output is rearranged back to mel by time format.

Model sizes and hyperparameters used in experiments:

- latent dimension equal to 256,
- encoder projection dimension equal to 256,  
- decoder hidden size equal to 256,  
- encoder GRU layers equal to 2, 
- decoder GRU layers equal to 2.

---

## Training objective and optimisation

Reconstruction loss is the elementwise mean absolute error in mel dB space. The training settings used in the notebook:

- batch size equal to 32  
- number of epochs equal to 50
- optimizer Adam with learning rate equal to 1e-3 and betas equal to 0.5 and 0.999  
- gradient clipping norm equal to 1.0  
- model checkpoints saved after every epoch.

All training is performed on T4 GPU in Google Colab.

---

## Results and evaluation

At each epoch, the training L1 loss was saved to the history for further visualization. The presented graph shows a high stability of the learning process.

<img width="500" height="240" alt="Screenshot 2025-12-31 232235" src="https://github.com/user-attachments/assets/ba8ce165-8729-45cf-af21-b327e473971b" />

The final L1 loss is approximately 0.17, which is sufficient to generate realistic spectrograms.
