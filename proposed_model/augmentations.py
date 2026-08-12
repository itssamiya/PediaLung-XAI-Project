import numpy as np


class SpectrogramAugmentation:

    def __init__(
        self,
        max_time_shift=10,
        time_mask_width=10,
        freq_mask_width=8,
        noise_std=0.005,
    ):

        self.max_time_shift = max_time_shift
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.noise_std = noise_std

    ############################################################
    # Random Time Shift
    ############################################################

    def time_shift(self, x):

        shift = np.random.randint(
            -self.max_time_shift,
            self.max_time_shift + 1,
        )

        return np.roll(x, shift, axis=-1)

    ############################################################
    # Time Mask
    ############################################################

    def time_mask(self, x):

        width = np.random.randint(
            4,
            self.time_mask_width + 1,
        )

        t = x.shape[-1]

        if width >= t:
            return x

        start = np.random.randint(0, t - width)

        x[..., start : start + width] = 0

        return x

    ############################################################
    # Frequency Mask
    ############################################################

    def frequency_mask(self, x):

        width = np.random.randint(
            3,
            self.freq_mask_width + 1,
        )

        f = x.shape[-2]

        if width >= f:
            return x

        start = np.random.randint(0, f - width)

        x[..., start : start + width, :] = 0

        return x

    ############################################################
    # Gaussian Noise
    ############################################################

    def add_noise(self, x):

        noise = np.random.normal(
            0,
            self.noise_std,
            x.shape,
        )

        return x + noise

    ############################################################
    # Complete Augmentation Pipeline
    ############################################################

    def __call__(self, x):

        x = self.time_shift(x)

        if np.random.rand() < 0.5:
            x = self.time_mask(x)

        if np.random.rand() < 0.5:
            x = self.frequency_mask(x)

        if np.random.rand() < 0.5:
            x = self.add_noise(x)

        return x
