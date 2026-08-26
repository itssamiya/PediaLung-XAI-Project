import pandas as pd
from sklearn.preprocessing import LabelEncoder

from multifeature_dataset import MultiFeatureDataset

df = pd.read_csv("features/labels.csv")

encoder = LabelEncoder()

df["label_encoded"] = encoder.fit_transform(df["label"])

dataset = MultiFeatureDataset(
    df,
    "features"
)

mfcc, mel, chroma, label = dataset[0]

print("MFCC :", mfcc.shape)
print("Mel :", mel.shape)
print("Chroma :", chroma.shape)
print("Label :", label)