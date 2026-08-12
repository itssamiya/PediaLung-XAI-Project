import os
import pandas as pd

RESULTS_DIR = "results"

df = pd.read_csv(
    os.path.join(
        RESULTS_DIR,
        "experiment_comparison.csv",
    )
)

print("=" * 70)
print("PEDIALUNG-XAI RESEARCH DASHBOARD")
print("=" * 70)

print("\nAll Experiments\n")
print(df)

print("\n")

metrics = [
    "Accuracy",
    "Weighted_F1",
    "Macro_F1",
    "Balanced_Accuracy",
]

for metric in metrics:

    best = df.loc[df[metric].idxmax()]

    print("-" * 70)

    print(metric)

    print(f"Best Experiment : {best['Experiment']}")

    print(f"Score           : {best[metric]:.4f}")

print("-" * 70)

print("\nDashboard Completed.")
