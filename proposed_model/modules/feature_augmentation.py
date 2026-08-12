import random
import numpy as np


class FeatureAugmentation:

    def __init__(self):

        self.time_mask_width = 8
        self.freq_mask_width = 5
        self.noise_std = 0.005

    #######################################################
    # Frequency Mask
    #######################################################

    def frequency_mask(self, feature):

        feature = feature.copy()

        num_bins = feature.shape[0]

        width = random.randint(0, self.freq_mask_width)

        if width == 0:
            return feature

        start = random.randint(0, max(0, num_bins - width))

        feature[start : start + width, :] = 0

        return feature

    #######################################################
    # Time Mask
    #######################################################

    def time_mask(self, feature):

        feature = feature.copy()

        num_frames = feature.shape[1]

        width = random.randint(0, self.time_mask_width)

        if width == 0:
            return feature

        start = random.randint(0, max(0, num_frames - width))

        feature[:, start : start + width] = 0

        return feature

    #######################################################
    # Gaussian Noise
    #######################################################

    def gaussian_noise(self, feature):

        noise = np.random.normal(
            0,
            self.noise_std,
            feature.shape,
        )

        return feature + noise

    #######################################################
    # Apply
    #######################################################

    def __call__(self, feature):

        if random.random() < 0.5:
            feature = self.frequency_mask(feature)

        if random.random() < 0.5:
            feature = self.time_mask(feature)

        if random.random() < 0.5:
            feature = self.gaussian_noise(feature)

        return feature.astype(np.float32)
