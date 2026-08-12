import librosa


def extract_mfcc(signal, sr, n_mfcc=40):
    """
    Extract MFCC features exactly like Paper 1.
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    return mfcc