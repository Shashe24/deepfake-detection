"""
Configuration file for deepfake detection project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration - Updated for Kaggle dataset
KAGGLE_DATASET_PATH = Path(r"C:\Users\shash\.cache\kagglehub\datasets\saurabhbagchi\deepfake-image-detection\versions\2")
TRAIN_DIR = KAGGLE_DATASET_PATH / "train-20250112T065955Z-001" / "train"
TEST_DIR = KAGGLE_DATASET_PATH / "test-20250112T065939Z-001" / "test"
REAL_DIR = DATA_DIR / "real"  # Keep for backward compatibility
FAKE_DIR = DATA_DIR / "fake"  # Keep for backward compatibility
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

# Data preprocessing - Optimized for faster training
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced for CPU training
NUM_WORKERS = 0  # Disabled to avoid multiprocessing issues

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Training configuration - Optimized for faster training
EPOCHS = 20  # Reduced epochs for faster training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Reduced patience for faster convergence

# Model configuration - Optimized for faster training
MODEL_CONFIGS = {
    'efficientnet': {
        'model_name': 'efficientnet-b0',  # Smaller model for faster training
        'pretrained': True,
        'num_classes': 2
    },
    'xception': {
        'pretrained': True,
        'num_classes': 2
    },
    'hybrid_ensemble': {
        'cnn_backbone': 'efficientnet-b0',  # Smaller backbone
        'transformer_model': 'google/vit-base-patch16-224',
        'num_classes': 2
    }
}

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_limit': 20,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'saturation_limit': 0.2,
    'hue_shift_limit': 10,
    'noise_var_limit': (10.0, 50.0),
    'blur_limit': 3,
    'p': 0.5  # Probability of applying augmentation
}

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']

# Deployment configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = MODELS_DIR / "best_model.pth"

# Input normalization (standardized across all models)
INPUT_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'input_range': [0, 1]  # Normalized to [0, 1] range
}

# File validation settings
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE_MB = 16
MIN_IMAGE_SIZE = (32, 32)
MAX_IMAGE_SIZE = (4096, 4096)

# Model validation settings
MIN_MODEL_FILE_SIZE_KB = 1
MODEL_VALIDATION_INPUT_SHAPE = (1, 3, 224, 224)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Demo model configuration
DEMO_MODEL_CONFIG = {
    'model_name': 'efficientnet-b0',
    'num_classes': 2,
    'pretrained': False,
    'dropout_rate': 0.3,
    'weight_init_gain': 0.1,
    'bias_init_std': 0.01,
    'noise_std': 0.001
}
