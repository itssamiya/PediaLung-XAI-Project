import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from hierarchical_predictor import HierarchicalPredictor

print("=" * 60)
print("PediaLung-XAI DEPLOYMENT TEST")
print("=" * 60)


predictor = HierarchicalPredictor()


# ------------------------------------------------------------
# Dummy feature shapes matching the trained system
# ------------------------------------------------------------

mfcc = np.random.randn(40, 94).astype(np.float32)

mel = np.random.randn(128, 259).astype(np.float32)

chroma = np.random.randn(12, 259).astype(np.float32)


result = predictor.predict_features(mfcc, mel, chroma)


print("\nPrediction:")
print(result["prediction"])

print("\nConfidence:")
print(f"{result['confidence'] * 100:.2f}%")

print("\nNormal probability:")
print(f"{result['normal_probability'] * 100:.2f}%")

print("\nAbnormal probability:")
print(f"{result['abnormal_probability'] * 100:.2f}%")

print("\nTop 3:")

for item in result["top3"]:

    print(f"{item['class']}: " f"{item['probability'] * 100:.2f}%")

print("\nProbability sum:")

print(sum(result["probabilities"].values()))

print("\nInference time:")

print(f"{result['inference_time_ms']:.2f} ms")

print("\n" + "=" * 60)
print("DEPLOYMENT TEST FINISHED")
print("=" * 60)
