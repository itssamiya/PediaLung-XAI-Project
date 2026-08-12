import pandas as pd
import matplotlib.pyplot as plt
import os

from config import EXPERIMENT_NAME
from config import SAVE_DIR
from config import MODEL_DIR

history = pd.read_csv(os.path.join(SAVE_DIR, "history.csv"))


# Accuracy

plt.figure(figsize=(7, 5))

plt.plot(history["epoch"], history["train_acc"], label="Training Accuracy", linewidth=2)

plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy", linewidth=2)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "accuracy_curve.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Loss

plt.figure(figsize=(7, 5))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss",
    linewidth=2,
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    linewidth=2,
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "loss_curve.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()
