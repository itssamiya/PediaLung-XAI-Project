import numpy as np

data = np.load(
    "features/paper1_mfcc_features.npz",
    allow_pickle=True
)

mfcc = data["mfcc"]
labels = data["labels"]
filenames = data["filenames"]

print("Total Samples :", len(labels))
print("Unique Labels :", np.unique(labels))
print("First Sample Shape :", mfcc[0].shape)
print("First Label :", labels[0])
print("First Filename :", filenames[0])