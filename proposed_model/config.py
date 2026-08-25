import os

# ==========================================
# Experiment Settings
# ==========================================

EXPERIMENT_NAME = "proposed_focal_sampler"

MODEL_NAME = "proposed"


# ==========================================
# Model Configuration
# ==========================================

MODEL_CONFIG = {
    "baseline": {
        "use_residual": False,
        "use_se": False,
        "use_fusion": False,
        "use_attention": False,
    },
    "residual": {
        "use_residual": True,
        "use_se": False,
        "use_fusion": False,
        "use_attention": False,
    },
    "residual_se": {
        "use_residual": True,
        "use_se": True,
        "use_fusion": False,
        "use_attention": False,
    },
    "fusion": {
        "use_residual": True,
        "use_se": True,
        "use_fusion": True,
        "use_attention": False,
    },
    "proposed": {
        "use_residual": True,
        "use_se": True,
        "use_fusion": True,
        "use_attention": True,
    },
}

# ==========================================
# Directories
# ==========================================

RESULTS_ROOT = "results"

MODEL_DIR = "saved_models"

SAVE_DIR = os.path.join(
    RESULTS_ROOT,
    EXPERIMENT_NAME,
)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# Training Configuration
# ==========================================

BATCH_SIZE = 16

NUM_EPOCHS = 40

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.1

FOCAL_GAMMA = 2.0

EARLY_STOPPING_PATIENCE = 10

USE_WEIGHTED_SAMPLER = True
