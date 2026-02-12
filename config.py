"""
BIPA (Biologically-Inspired Patch Attention) Configuration File
================================================================
All hyperparameters, paths, and settings for the BIPA model.
Modify this file to change dataset, model architecture, or training parameters.
"""

import os
import torch

# ====================================================================
# DATASET CONFIGURATION
# ====================================================================

# Select which dataset to use (1, 2, 3, or 4)
ACTIVE_DATASET = 1

# Dataset paths (modify these based on your setup)
DATASET_PATHS = {
    1: "/content/drive/MyDrive/Dataset_1",  # Small-scale imbalanced (442 images)
    2: "/content/drive/MyDrive/Dataset_2",  # Medium-scale imbalanced (2,640 images)
    3: "/content/drive/MyDrive/Dataset_3",  # Balanced large-scale (2,400 images)
    4: "/content/drive/MyDrive/Dataset_4",  # Large-scale complex (5,482 images)
}

# For local setup, uncomment and modify these paths:
# DATASET_PATHS = {
#     1: "./data/Dataset_1",
#     2: "./data/Dataset_2",
#     3: "./data/Dataset_3",
#     4: "./data/Dataset_4",
# }

# Active dataset root
DRIVE_ROOT = DATASET_PATHS[ACTIVE_DATASET]

# CSV file paths
TRAIN_CSV = os.path.join(DRIVE_ROOT, f"d{ACTIVE_DATASET}_train.csv")
TEST_CSV = os.path.join(DRIVE_ROOT, f"d{ACTIVE_DATASET}_test.csv")

# Output directories
CROP_DIR = os.path.join(DRIVE_ROOT, "crops_yolo")
BAM_CACHE_DIR = os.path.join(DRIVE_ROOT, "bam_cache")
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")

# Processed CSV paths
CROPPED_TRAIN_CSV = os.path.join(CROP_DIR, f"d{ACTIVE_DATASET}_train_crops.csv")
CROPPED_TEST_CSV = os.path.join(CROP_DIR, f"d{ACTIVE_DATASET}_test_crops.csv")

# Model save paths
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "BIPA_best.pth")
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "BIPA_final.pth")

# ====================================================================
# MODEL ARCHITECTURE PARAMETERS
# ====================================================================

# Image and patch settings
IMG_SIZE = 224          # Input image size (224x224)
PATCH_SIZE = 8          # Micro-patch size (8x8 pixels as per paper)
IN_CHANNELS = 3         # RGB images

# Transformer architecture
EMBED_DIM = 128         # Embedding dimension (d_model)
DEPTH = 4               # Number of BIPA encoder blocks (L)
NUM_HEADS = 8           # Number of attention heads
MLP_RATIO = 4.0         # MLP hidden dimension ratio
DROP_RATE = 0.1         # Dropout rate

# A-BAMNet architecture
ABAM_BASE_CHANNELS = 32  # Base channels for A-BAMNet CNN

# Derived values (don't modify)
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Total number of patches

# ====================================================================
# BAM (BIOLOGICAL ATTENTION MAP) CONFIGURATION
# ====================================================================

# BAM Generation Modes:
# Mode 1: Heuristic only (bio-inspired filters: Gabor, LoG, Contours)
# Mode 2: Learned only (pure A-BAMNet, no heuristic initialization)
# Mode 3: Heuristic + Learned (A-BAMNet uses heuristic as additional input)

BAM_MODE = "heuristic+learned"  # Options: "heuristic", "learned", "heuristic+learned"

# Pre-computation settings
PRECOMPUTE_HEURISTIC_BAMS = True   # Pre-compute and cache heuristic BAMs (10-20x faster)
USE_HEURISTIC_BAMS = (BAM_MODE in ["heuristic", "heuristic+learned"])

# Heuristic BAM filter parameters
GABOR_FREQUENCIES = [0.1, 0.2]     # Gabor filter frequencies
GABOR_ORIENTATIONS = 4             # Number of Gabor orientations
LOG_SIGMA = 2.0                    # Laplacian of Gaussian sigma
CANNY_THRESHOLD_LOW = 50           # Canny edge detection low threshold
CANNY_THRESHOLD_HIGH = 150         # Canny edge detection high threshold

# BAM aggregation weights
BAM_GABOR_WEIGHT = 0.5             # Weight for Gabor texture features
BAM_LOG_WEIGHT = 0.3               # Weight for LoG edge features
BAM_CONTOUR_WEIGHT = 0.2           # Weight for contour features

# ====================================================================
# TRAINING HYPERPARAMETERS
# ====================================================================

# Training settings
EPOCHS = 30                        # Total training epochs
BATCH_SIZE = 32                    # Batch size (increased from 16 for speed)
LEARNING_RATE = 1e-4               # Initial learning rate
WEIGHT_DECAY = 0.01                # AdamW weight decay
NUM_WORKERS = 4                    # DataLoader workers (increase for speed)

# Learning rate scheduler
LR_SCHEDULER = "cosine"            # Options: "cosine", "step", "none"
WARMUP_EPOCHS = 0                  # Number of warmup epochs

# Checkpointing
SAVE_CHECKPOINT_EVERY = 5          # Save checkpoint every N epochs
KEEP_BEST_N_CHECKPOINTS = 3        # Keep top N best checkpoints

# Early stopping
EARLY_STOPPING_PATIENCE = 10       # Stop if no improvement for N epochs
EARLY_STOPPING_METRIC = "f1"       # Metric to monitor: "accuracy", "f1", "loss"

# ====================================================================
# DATA AUGMENTATION
# ====================================================================

# Training augmentation
TRAIN_AUG_HORIZONTAL_FLIP = 0.5    # Probability of horizontal flip
TRAIN_AUG_ROTATION = 15            # Rotation range in degrees
TRAIN_AUG_BRIGHTNESS = 0.2         # Brightness jitter
TRAIN_AUG_CONTRAST = 0.2           # Contrast jitter
TRAIN_AUG_SATURATION = 0.2         # Saturation jitter

# Normalization (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ====================================================================
# YOLO PREPROCESSING
# ====================================================================

YOLO_MODEL = "yolov8n.pt"          # YOLOv8 model variant
YOLO_CONFIDENCE_THRESHOLD = 0.25   # Detection confidence threshold
YOLO_PADDING_RATIO = 0.1           # Padding around detected bounding box
CENTER_CROP_RATIO = 0.9            # Fallback center crop ratio if no detection

# ====================================================================
# EVALUATION SETTINGS
# ====================================================================

# Metrics to compute
COMPUTE_CONFUSION_MATRIX = True
COMPUTE_PER_CLASS_METRICS = True
COMPUTE_INFERENCE_TIME = True

# Visualization
VIZ_NUM_SAMPLES = 6                # Number of samples for BAM visualization
VIZ_SAMPLES_PER_CLASS = 2          # Samples per class for visualization

# ====================================================================
# HARDWARE & PERFORMANCE
# ====================================================================

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mixed precision training (for faster training on modern GPUs)
USE_AMP = True if torch.cuda.is_available() else False

# Reproducibility
RANDOM_SEED = 42                   # Random seed for reproducibility
DETERMINISTIC = False              # Set True for fully deterministic training (slower)

# ====================================================================
# LOGGING & VERBOSITY
# ====================================================================

VERBOSE = True                     # Print detailed logs
LOG_INTERVAL = 10                  # Log every N batches during training
SAVE_TRAINING_PLOTS = True         # Save training history plots
SAVE_BAM_VISUALIZATIONS = True     # Save BAM attention visualizations

# ====================================================================
# EXPECTED DATASET PERFORMANCE (from paper)
# ====================================================================

EXPECTED_RESULTS = {
    1: {"accuracy": 48.31, "f1": 0.4667},   # Dataset-1: Small imbalanced
    2: {"accuracy": 76.52, "f1": 0.7023},   # Dataset-2: Medium imbalanced (CORRECTED)
    3: {"accuracy": 99.67, "f1": 0.9967},   # Dataset-3: Balanced
    4: {"accuracy": 88.06, "f1": 0.8763},   # Dataset-4: Large complex
}

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_dataset_info():
    """Get information about the active dataset"""
    dataset_info = {
        1: {"name": "Dataset-1", "size": 442, "balance": "Imbalanced", "description": "Small-scale imbalanced"},
        2: {"name": "Dataset-2", "size": 2640, "balance": "Imbalanced", "description": "Medium-scale imbalanced"},
        3: {"name": "Dataset-3", "size": 2400, "balance": "Balanced", "description": "Balanced large-scale"},
        4: {"name": "Dataset-4", "size": 5482, "balance": "Imbalanced", "description": "Large-scale complex"},
    }
    return dataset_info.get(ACTIVE_DATASET, {})

def validate_config():
    """Validate configuration parameters"""
    assert ACTIVE_DATASET in [1, 2, 3, 4], f"ACTIVE_DATASET must be 1, 2, 3, or 4, got {ACTIVE_DATASET}"
    assert PATCH_SIZE > 0 and IMG_SIZE % PATCH_SIZE == 0, "IMG_SIZE must be divisible by PATCH_SIZE"
    assert EMBED_DIM % NUM_HEADS == 0, "EMBED_DIM must be divisible by NUM_HEADS"
    assert BAM_MODE in ["heuristic", "learned", "heuristic+learned"], f"Invalid BAM_MODE: {BAM_MODE}"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert EPOCHS > 0, "EPOCHS must be positive"
    
    # Create directories if they don't exist
    for directory in [CROP_DIR, BAM_CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    return True

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("BIPA CONFIGURATION")
    print("=" * 70)
    print(f"\n📁 Dataset:")
    print(f"   Active Dataset: Dataset-{ACTIVE_DATASET}")
    dataset_info = get_dataset_info()
    if dataset_info:
        print(f"   Description: {dataset_info['description']}")
        print(f"   Total Images: {dataset_info['size']}")
        print(f"   Balance: {dataset_info['balance']}")
    print(f"   Path: {DRIVE_ROOT}")
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"   Number of Patches: {NUM_PATCHES}")
    print(f"   Embedding Dim: {EMBED_DIM}")
    print(f"   Depth: {DEPTH} blocks")
    print(f"   Attention Heads: {NUM_HEADS}")
    print(f"   MLP Ratio: {MLP_RATIO}")
    
    print(f"\n🔬 BAM Configuration:")
    print(f"   Mode: {BAM_MODE}")
    print(f"   Pre-compute Heuristic: {PRECOMPUTE_HEURISTIC_BAMS}")
    print(f"   Use Heuristic BAMs: {USE_HEURISTIC_BAMS}")
    
    print(f"\n🎯 Training:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    print(f"   Device: {DEVICE}")
    print(f"   Mixed Precision: {USE_AMP}")
    
    print(f"\n📊 Expected Performance (from paper):")
    expected = EXPECTED_RESULTS.get(ACTIVE_DATASET, {})
    if expected:
        print(f"   Accuracy: {expected['accuracy']}%")
        print(f"   F1-Score: {expected['f1']}")
    
    print("=" * 70)

# Validate configuration on import
if __name__ != "__main__":
    validate_config()
