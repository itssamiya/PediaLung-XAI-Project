import os


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

import librosa
from skimage.restoration import denoise_wavelet


DATASET_PATH = Path(r"D:\Research\SPRSound-main\SPRSound-main\BioCAS2022")


def load_one_record():

    wav_folder = DATASET_PATH / "train2022_wav"
    json_folder = DATASET_PATH / "train2022_json"

    # Find the first recording with respiratory events
    json_files = sorted(json_folder.glob("*.json"))

    json_file = None
    annotation = None

    for file in json_files:

        with open(file, "r") as f:
            ann = json.load(f)

        if len(ann.get("event_annotation", [])) > 0:
            json_file = file
            annotation = ann
            break

    if json_file is None:
        raise RuntimeError("No annotated recordings found.")

    wav_file = wav_folder / (json_file.stem + ".wav")

    print("=" * 60)
    print("JSON :", json_file.name)
    print("WAV  :", wav_file.name)
    print("=" * 60)

    print("\nEntire JSON:")
    print(annotation)

    print("\nNumber of events:", len(annotation["event_annotation"]))

   
    signal, sr = librosa.load(wav_file, sr=None)

    duration = len(signal) / sr

    print(f"\nSampling Rate : {sr}")
    print(f"Duration      : {duration:.2f} seconds")

    print("\nRecord Label:")
    print(annotation.get("record_annotation", "Not Available"))

    print("\nRespiratory Events")
    print("-" * 60)

  

    for i, event in enumerate(annotation["event_annotation"], start=1):

        start = int(event["start"])
        end = int(event["end"])
        label = event["type"]

        print(f"{i:02d}. {label:<18} {start} ms -> {end} ms")

        start_sample = int(start * sr / 1000)
        end_sample = int(end * sr / 1000)

      
        event_signal = signal[start_sample:end_sample]

        print(f"Original Samples : {len(event_signal)}")


        denoised_signal = denoise_wavelet(
            event_signal,
            method="BayesShrink",
            mode="soft",
            wavelet="sym8",
            wavelet_levels=3,
            rescale_sigma=True
        )

        print("Wavelet Denoising Completed!")

      

        TARGET_DURATION = 6
        TARGET_SAMPLES = TARGET_DURATION * sr

        while len(denoised_signal) < TARGET_SAMPLES:
            denoised_signal = librosa.util.fix_length(
                denoised_signal,
                size=len(denoised_signal) * 2
            )

        denoised_signal = denoised_signal[:TARGET_SAMPLES]

        print(f"Final Samples : {len(denoised_signal)}")
        print(f"Final Duration: {len(denoised_signal)/sr:.2f} sec")

      

        mfcc = librosa.feature.mfcc(
            y=denoised_signal,
            sr=sr,
            n_mfcc=40
        )

        print("MFCC Shape :", mfcc.shape)

        print("-" * 60)


if __name__ == "__main__":
    load_one_record()