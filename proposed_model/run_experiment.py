import subprocess
import sys

steps = [
    "train_proposed_model.py",
    "plot_history.py",
    "evaluate_proposed_model.py",
    "gradcam.py",
]

for script in steps:

    print("\n" + "=" * 60)
    print(f"Running {script}")
    print("=" * 60)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:

        print(f"\nError while running {script}")

        break

print("\nExperiment Finished.")
