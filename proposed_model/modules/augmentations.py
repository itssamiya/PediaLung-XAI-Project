import torch
import random


class SpecAugment:

    def __init__(
        self,
        freq_mask_param=12,
        time_mask_param=30,
        num_freq_masks=2,
        num_time_masks=2,
    ):

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec):

        spec = spec.clone()

        # -----------------------------
        # Frequency Mask
        # -----------------------------
        for _ in range(self.num_freq_masks):

            freq = random.randint(0, self.freq_mask_param)

            if freq == 0:
                continue

            f0 = random.randint(0, max(0, spec.shape[1] - freq))

            spec[:, f0 : f0 + freq, :] = 0

        # -----------------------------
        # Time Mask
        # -----------------------------
        for _ in range(self.num_time_masks):

            time = random.randint(0, self.time_mask_param)

            if time == 0:
                continue

            t0 = random.randint(0, max(0, spec.shape[2] - time))

            spec[:, :, t0 : t0 + time] = 0

        return spec
