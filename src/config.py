# src/config.py

# # ─── Data Path ───────────────────────────────────────────────────────────────
# DATA_PATH = "data/creditcard.csv"

# # ─── Preprocessing ──────────────────────────────────────────────────────────
# PCA_COMPONENTS = 4

# # ─── Reproducibility ─────────────────────────────────────────────────────────
# RANDOM_SEED = 42

# # ─── Classical Model Hyperparameters ─────────────────────────────────────────
# RF_N_ESTIMATORS = 100

# # ─── Quantum Model Hyperparameters ───────────────────────────────────────────
# # PCA_COMPONENTS = 4
# # QML_FEATURE_MAP_REPS = 2      # for QSVM
# # VQC_LAYERS = 2                # for VQC & QNN ansatz depth
# # VQC_MAXITER = 100
# # QNN_LEARNING_RATE = 0.01
# # QNN_EPOCHS = 10

# PCA_COMPONENTS = 8
# VQC_LAYERS = 6            # Increased from 2 or 3
# VQC_MAXITER = 100            
# QNN_EPOCHS = 50           # More training steps
# QNN_LEARNING_RATE = 0.01

# # For training control
# SEED = 42
# EARLY_STOPPING_PATIENCE = 10  # Number of epochs with no improvement to wait
# LR_PATIENCE = 3              # Reduce LR if no improvement for this many epochs
# LR_FACTOR = 0.5              # Learning rate reduction factor
# LOG_FILE = "training_log.txt"

# src/config.py

# ─── Data Path ───────────────────────────────────────────────────────────────
DATA_PATH = "data/creditcard.csv"

# ─── Preprocessing ──────────────────────────────────────────────────────────
# Using 8 PCA components for increased feature representation
PCA_COMPONENTS = 8

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Classical Model Hyperparameters ─────────────────────────────────────────
RF_N_ESTIMATORS = 100

# ─── Quantum Model Hyperparameters ───────────────────────────────────────────
VQC_LAYERS = 6            # Increased for deeper quantum circuits
VQC_MAXITER = 100            
QNN_EPOCHS = 50           # More training epochs for better convergence
QNN_LEARNING_RATE = 0.05

# ─── Training Control ────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10  # Number of epochs with no improvement to wait
LR_PATIENCE = 3               # Reduce LR if no improvement for this many epochs
LR_FACTOR = 0.5               # Learning rate reduction factor
LOG_FILE = "training_log.txt"
