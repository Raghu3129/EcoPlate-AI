"""
Training script for the EcoPlate AI Random Forest waste predictor.

Trains two models:
  1. A RandomForestClassifier predicting the categorical waste_level.
  2. A RandomForestRegressor predicting the continuous
     waste_percentage.

Both share the same scaled feature set. Artifacts (models, scaler,
label encoder) are persisted with joblib for later loading by the
prediction service.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Allow running as a standalone script (python app/ml/train_model.py)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.config import (  # noqa: E402
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    SYNTHETIC_DATASET_PATH,
)
from app.ml.synthetic_data_generator import (  # noqa: E402
    generate_synthetic_dataset,
    save_synthetic_dataset,
)

FEATURE_COLUMNS = [
    "plate_coverage_ratio",
    "leftover_area_px",
    "total_plate_area_px",
    "color_variance",
    "texture_score",
    "brightness_mean",
]


def load_or_create_dataset() -> pd.DataFrame:
    """Load the synthetic dataset from disk, generating it if missing."""
    if SYNTHETIC_DATASET_PATH.exists():
        return pd.read_csv(SYNTHETIC_DATASET_PATH)
    dataset = generate_synthetic_dataset()
    save_synthetic_dataset(dataset)
    return dataset


def train_and_save_models() -> dict:
    """
    Train the classification and regression Random Forest models and
    persist all artifacts to disk.

    Returns:
        A dictionary of evaluation metrics for both models.
    """
    dataset = load_or_create_dataset()

    x = dataset[FEATURE_COLUMNS].values
    y_level = dataset["waste_level"].values
    y_percentage = dataset["waste_percentage"].values

    label_encoder = LabelEncoder()
    y_level_encoded = label_encoder.fit_transform(y_level)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    (
        x_train,
        x_test,
        y_level_train,
        y_level_test,
        y_pct_train,
        y_pct_test,
    ) = train_test_split(
        x_scaled,
        y_level_encoded,
        y_percentage,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_level_encoded,
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_level_train)

    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    regressor.fit(x_train, y_pct_train)

    level_predictions = classifier.predict(x_test)
    percentage_predictions = regressor.predict(x_test)

    metrics = {
        "classification_accuracy": float(accuracy_score(y_level_test, level_predictions)),
        "regression_mae": float(mean_absolute_error(y_pct_test, percentage_predictions)),
        "regression_r2": float(r2_score(y_pct_test, percentage_predictions)),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }

    # Persist artifacts: a single bundle containing both models for
    # simpler loading downstream.
    model_bundle = {"classifier": classifier, "regressor": regressor}
    joblib.dump(model_bundle, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return metrics


if __name__ == "__main__":
    results = train_and_save_models()
    print("Training complete. Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value}")
