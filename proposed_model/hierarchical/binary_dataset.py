import torch
from torch.utils.data import Dataset


class BinaryMultiFeatureDataset(Dataset):

    def __init__(self, dataframe, feature_root="features", train=False):

        self.df = dataframe.reset_index(drop=True)
        self.feature_root = feature_root
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        mfcc_path = row["mfcc_path"]
        mel_path = row["mel_path"]
        chroma_path = row["chroma_path"]

        mfcc = torch.tensor(
            __import__("numpy").load(
                f"{self.feature_root}/{mfcc_path}"
            ),
            dtype=torch.float32,
        )

        mel = torch.tensor(
            __import__("numpy").load(
                f"{self.feature_root}/{mel_path}"
            ),
            dtype=torch.float32,
        )

        chroma = torch.tensor(
            __import__("numpy").load(
                f"{self.feature_root}/{chroma_path}"
            ),
            dtype=torch.float32,
        )

        # Normal = 0
        # Abnormal = 1
        label = int(row["binary_encoded"])

        return (
            mfcc.unsqueeze(0),
            mel.unsqueeze(0),
            chroma.unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )