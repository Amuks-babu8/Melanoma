# Dimension of resized image
import os

DEFAULT_IMAGE_SIZE = tuple((256, 256))

# images to train with
N_IMAGES = 30
# dataset folder
ROOT_DIR = "Dataset"

# training dataset folder
TRAIN_DIR = os.path.join(ROOT_DIR, "Training_Dataset")

# Testing dataset folder
VAL_DIR = os.path.join(ROOT_DIR, "Testing_Dataset")

TRAIN_EPOCHS = 10
TRAIN_STEPS = 150
LR = 1e-3
BATCH_SIZE = 32
IMG_WIDTH = 256
IMG_HEIGHT = 256
DEPTH = 3

# where to save labels
LABELS_FILE = "melanoma_labels.pkl"

# where to save models
MODEL_FILE = "melanoma.h5"
