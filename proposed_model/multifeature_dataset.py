import os
import numpy as np
import torch
from torch.utils.data import Dataset
from modules.feature_augmentation import FeatureAugmentation
from modules.augmentations import SpecAugment
from augmentations import SpectrogramAugmentation


class MultiFeatureDataset(Dataset):

    def __init__(
        self,
        dataframe,
        feature_root,
        train=False,
    ):

        self.specaugment = SpecAugment()
        self.dataframe = dataframe.reset_index(drop=True)

        self.train = train
        self.augment = FeatureAugmentation()

        self.mfcc_dir = os.path.join(feature_root, "mfcc")
        self.mel_dir = os.path.join(feature_root, "mel")
        self.chroma_dir = os.path.join(feature_root, "chroma")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        filename = row["filename"]

        label = row["label_encoded"]

        mfcc = np.load(os.path.join(self.mfcc_dir, filename))

        mel = np.load(os.path.join(self.mel_dir, filename))

        chroma = np.load(os.path.join(self.chroma_dir, filename))

        if self.train:

            mfcc = self.augment(mfcc)

            chroma = self.augment(chroma)

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        if self.train:
            mel = self.specaugment(mel)

        chroma = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return mfcc, mel, chroma, label
