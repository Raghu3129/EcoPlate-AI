"""
Central configuration for the EcoPlate AI backend.

All environment-dependent and tunable constants live here so the rest
of the codebase can import a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "trained_models"
UPLOAD_DIR: Path = BASE_DIR / "uploads"
EXPORT_DIR: Path = BASE_DIR / "exports"

for directory in (DATA_DIR, MODEL_DIR, UPLOAD_DIR, EXPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'ecoplate.db').as_posix()}"
)

# ---------------------------------------------------------------------------
# ML model artifacts
# ---------------------------------------------------------------------------
MODEL_PATH: Path = MODEL_DIR / "waste_predictor.joblib"
SCALER_PATH: Path = MODEL_DIR / "feature_scaler.joblib"
LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.joblib"
SYNTHETIC_DATASET_PATH: Path = DATA_DIR / "synthetic_dataset.csv"

# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
IMAGE_RESIZE_DIM: tuple = (256, 256)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_SIZE_MB: int = 10

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
FOOD_CATEGORIES = [
    "rice",
    "curry",
    "vegetables",
    "bread",
    "dal",
    "salad",
    "dessert",
    "mixed",
]

WASTE_LEVELS = ["low", "medium", "high", "severe"]

# Approximate plate diameter in cm used to convert pixel area to a
# physical scale reference (assumes a standard canteen plate).
REFERENCE_PLATE_DIAMETER_CM: float = 26.0

# Synthetic dataset generation
SYNTHETIC_SAMPLE_COUNT: int = 3200
RANDOM_SEED: int = 42

# API metadata
API_TITLE = "EcoPlate AI Backend"
API_DESCRIPTION = (
    "Canteen-Scale Micro Food Wastage Quantification Using Plate-Level "
    "Image Segmentation and Lightweight AI Inference."
)
API_VERSION = "1.0.0"
