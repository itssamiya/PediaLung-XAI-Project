import librosa
import numpy as np
from skimage.restoration import denoise_wavelet

TARGET_SR = 4000
TARGET_DURATION = 6


def load_audio(audio_path):

    signal, sr = librosa.load(audio_path, sr=TARGET_SR)

    return signal, sr


def wavelet_denoise(signal):

    signal = denoise_wavelet(
        signal, method="BayesShrink", mode="soft", rescale_sigma=True
    )

    return signal


def normalize_length(signal, sr):

    target_length = TARGET_DURATION * sr

    if len(signal) < target_length:

        pad = target_length - len(signal)

        signal = np.pad(signal, (0, pad))

    else:

        signal = signal[:target_length]

    return signal


def extract_mfcc(signal, sr):

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    return mfcc.astype(np.float32)


def extract_mel(signal, sr):

    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)

    mel = librosa.power_to_db(mel)

    return mel.astype(np.float32)


def extract_chroma(signal, sr):

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)

    return chroma.astype(np.float32)
