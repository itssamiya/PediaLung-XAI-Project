import os
import subprocess
import sys

experiments = [
    "baseline",
    "residual",
    "residual_se",
    "fusion",
    "proposed",
]

for exp in experiments:

    print("\n")
    print("=" * 70)
    print(f"Running Experiment : {exp}")
    print("=" * 70)

    # ---------------------------------------
    # Update config.py automatically
    # ---------------------------------------

    with open("config.py", "r") as f:
        text = f.read()

    import re

    text = re.sub(
        r'EXPERIMENT_NAME\s*=\s*".*"',
        f'EXPERIMENT_NAME = "{exp}"',
        text,
    )

    with open("config.py", "w") as f:
        f.write(text)

    # ---------------------------------------
    # Run complete pipeline
    # ---------------------------------------

    result = subprocess.run([sys.executable, "run_experiment.py"])

    if result.returncode != 0:

        print(f"\nExperiment {exp} Failed.")

        break

print("\n")
print("=" * 70)
print("ALL EXPERIMENTS FINISHED")
print("=" * 70)
