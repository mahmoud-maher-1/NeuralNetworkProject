"""
Utils: Constants
"""

# --- App Config ---
APP_TITLE = "Neural Network Classifier Project"
APP_ICON = "👑"

# --- ML Config ---
RANDOM_SEED = 42
TRAIN_SAMPLES_PER_CLASS = 30
TEST_SAMPLES_PER_CLASS = 20

# --- Data Config ---
DATA_FILE_PATH = "data/dataset.csv"
PROCESSED_DATA_FILE_PATH = "data/processed.csv"
ALL_FEATURES = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]
ALL_CLASSES = ["A", "B", "C"]

# --- UI Defaults ---
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.01
DEFAULT_MSE_THRESH = 0.001