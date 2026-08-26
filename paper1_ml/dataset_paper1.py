import os
import numpy as np
import torch
from torch.utils.data import Dataset



class RespiratoryDataset(Dataset):

    def __init__(self, dataframe, feature_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        filename = row["filename"]

        label = row["label_encoded"]

        feature_path = os.path.join(self.feature_dir, filename)

        mfcc = np.load(feature_path)

        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        mfcc = mfcc.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return mfcc, label




