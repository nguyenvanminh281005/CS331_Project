"""
Configuration file for Age-Invariant Face Recognition project
"""
import os

# Project paths
PROJECT_ROOT = "/mnt/data/KHTN2023/CS331/project"
DATASET_ROOT = "/mnt/data/KHTN2023/CS331/dataset"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "saved_models")
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")

# Dataset selection (can be changed at runtime)
ACTIVE_DATASET = "agedb_30"  # Options: 'agedb_30', 'morph_2'

# Dataset configurations
AGEDB_CONFIG = {
    "name": "agedb_30",
    "image_folder": os.path.join(DATASET_ROOT, "agedb_30/agedb_30_112x112"),
    "annotation_file": os.path.join(DATASET_ROOT, "agedb_30/agedb_30_ann.txt"),
    "image_size": (112, 112),
    "num_classes": None,  # Will be determined from dataset
    "has_age_info": False,  # AgeDB-30 doesn't have explicit age in metadata
}

# MORPH-2 configurations
MORPH2_CONFIG = {
    "name": "morph_2",
    "dataset_root": os.path.join(DATASET_ROOT, "morph_2"),
    "train_csv": os.path.join(DATASET_ROOT, "morph_2/Index/Train.csv"),
    "val_csv": os.path.join(DATASET_ROOT, "morph_2/Index/Validation.csv"),
    "test_csv": os.path.join(DATASET_ROOT, "morph_2/Index/Test.csv"),
    "image_size": (112, 112),
    "num_classes": None,  # Will be determined from dataset
    "has_age_info": True,  # MORPH-2 has age, gender info
}

# Additional datasets (for AgeDB)
ADDITIONAL_AGEDB_DATASETS = {
    "lfw": {
        "image_folder": os.path.join(DATASET_ROOT, "agedb_30/lfw_112x112"),
        "annotation_file": os.path.join(DATASET_ROOT, "agedb_30/lfw_ann.txt"),
    },
    "calfw": {
        "image_folder": os.path.join(DATASET_ROOT, "agedb_30/calfw_112x112"),
        "annotation_file": os.path.join(DATASET_ROOT, "agedb_30/calfw_ann.txt"),
    },
    "cplfw": {
        "image_folder": os.path.join(DATASET_ROOT, "agedb_30/cplfw_112x112"),
        "annotation_file": os.path.join(DATASET_ROOT, "agedb_30/cplfw_ann.txt"),
    },
}

# Helper function to get active dataset config
def get_dataset_config():
    """Get configuration for the active dataset"""
    if ACTIVE_DATASET == "agedb_30":
        return AGEDB_CONFIG
    elif ACTIVE_DATASET == "morph_2":
        return MORPH2_CONFIG
    else:
        raise ValueError(f"Unknown dataset: {ACTIVE_DATASET}")

def set_active_dataset(dataset_name):
    """Set the active dataset"""
    global ACTIVE_DATASET
    if dataset_name not in ["agedb_30", "morph_2"]:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'agedb_30' or 'morph_2'")
    ACTIVE_DATASET = dataset_name
    print(f"Active dataset set to: {ACTIVE_DATASET}")

# Face detection configurations (for DeepFace)
FACE_DETECTOR_CONFIG = {
    "backend": "mtcnn",  # Options: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn'
    "min_face_size": 40,
    "align": True,
    "enforce_detection": True,
}

# Legacy MTCNN config (kept for reference, not used with DeepFace)
MTCNN_CONFIG = {
    "min_face_size": 40,
    "thresholds": [0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
    "factor": 0.709,
    "device": "cuda",
}

# Model configurations
MODEL_CONFIG = {
    "embedding_size": 512,
    "dropout": 0.0,
    "pretrained": True,
}

# ArcFace configurations
ARCFACE_CONFIG = {
    "margin": 0.5,  # m in ArcFace
    "scale": 64.0,  # s in ArcFace
    "embedding_size": 512,
}

# MagFace configurations
MAGFACE_CONFIG = {
    "l_a": 10,  # lower bound of feature magnitude
    "u_a": 110,  # upper bound of feature magnitude
    "l_margin": 0.45,
    "u_margin": 0.8,
    "scale": 64.0,
    "embedding_size": 512,
}

# Temporal-Aware Model configurations
TEMPORAL_CONFIG = {
    "alpha": 0.5,  # Weight for negative pair loss
    "margin": 0.4,  # Margin for contrastive learning
    "lambda_tcl": 0.3,  # Weight for temporal contrastive loss
    "time_gaps": [1, 2, 4, 6, 8, 10],  # Years for time-gap analysis
}

# Training configurations
TRAIN_CONFIG = {
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "lr_scheduler": "cosine",  # 'step', 'cosine', 'plateau'
    "warmup_epochs": 5,
    "num_workers": 8,
    "pin_memory": True,
    "mixed_precision": True,  # Use AMP for faster training
}

# Evaluation configurations
EVAL_CONFIG = {
    "batch_size": 256,
    "far_targets": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # FAR values for TAR calculation
    "distance_metric": "cosine",  # 'cosine' or 'euclidean'
    "threshold_range": (0.0, 1.0),
    "num_thresholds": 1000,
}

# Quality filtering configurations
QUALITY_CONFIG = {
    "min_quality_score": 0.3,  # Minimum face quality score (0-1)
    "use_magface_norm": True,  # Use MagFace feature magnitude as quality
    "min_magnitude": 10.0,  # Minimum feature magnitude
    "max_magnitude": 110.0,  # Maximum feature magnitude
}

# Pair generation configurations
PAIR_CONFIG = {
    "same_identity_pairs": 3000,  # Number of positive pairs per time gap
    "diff_identity_pairs": 3000,  # Number of negative pairs per time gap
    "min_time_gap": 1,  # Minimum years between images
    "max_time_gap": 10,  # Maximum years between images
    "time_gap_step": 1,  # Step size for time gap bins
}

# Visualization configurations
VIS_CONFIG = {
    "dpi": 300,
    "figsize": (10, 8),
    "color_palette": "Set2",
    "tsne_perplexity": 30,
    "tsne_n_iter": 1000,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
}

# Statistical analysis configurations
STATS_CONFIG = {
    "confidence_level": 0.95,
    "bootstrap_samples": 1000,
    "random_seed": 42,
}

# Device configuration
DEVICE = "cuda"  # or 'cpu'
SEED = 42

# Create necessary directories
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "features"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "pairs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "visualizations"), exist_ok=True)
