import numpy as np
from skimage.restoration import denoise_wavelet


def denoise_signal(signal):
    """
    Wavelet denoising used in Paper 1.
    """

    return denoise_wavelet(
        signal,
        method="BayesShrink",
        mode="soft",
        wavelet="sym8",
        wavelet_levels=3,
        rescale_sigma=True,
    )


def standardize_length(signal, sr, target_duration=6):
    """
    Repeat respiratory event until it reaches target duration,
    then trim to exact length.
    """

    target_samples = sr * target_duration

    repeat_times = int(np.ceil(target_samples / len(signal)))

    signal = np.tile(signal, repeat_times)

    signal = signal[:target_samples]

    return signal