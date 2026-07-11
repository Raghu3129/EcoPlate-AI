"""
Model loading and inference service.

Loads the persisted Random Forest classifier, regressor, scaler, and
label encoder into memory once (singleton pattern) and exposes a
simple `predict` interface used by the API layer.
"""

from typing import Optional

import joblib
import numpy as np

from app.config import LABEL_ENCODER_PATH, MODEL_PATH, SCALER_PATH
from app.ml.train_model import FEATURE_COLUMNS, train_and_save_models


class ModelNotAvailableError(Exception):
    """Raised when model artifacts cannot be loaded or trained."""


class WastePredictor:
    """
    Loads trained model artifacts and provides waste-level and
    waste-percentage predictions from a feature vector.

    Implemented as a lazily-initialized singleton so the (relatively
    expensive) model load happens only once per process.
    """

    _instance: Optional["WastePredictor"] = None

    def __init__(self) -> None:
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.label_encoder = None
        self._load_or_train()

    @classmethod
    def get_instance(cls) -> "WastePredictor":
        """Return the singleton WastePredictor instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_or_train(self) -> None:
        """Load model artifacts from disk, training them first if absent."""
        if not (MODEL_PATH.exists() and SCALER_PATH.exists() and LABEL_ENCODER_PATH.exists()):
            train_and_save_models()

        try:
            bundle = joblib.load(MODEL_PATH)
            self.classifier = bundle["classifier"]
            self.regressor = bundle["regressor"]
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailableError(
                f"Failed to load trained model artifacts: {exc}"
            ) from exc

    def predict(self, feature_dict: dict) -> dict:
        """
        Run inference on a single feature vector.

        Args:
            feature_dict: Dictionary containing all keys in
                FEATURE_COLUMNS.

        Returns:
            Dictionary with predicted_waste_level, predicted_waste_percentage,
            and confidence_score.

        Raises:
            ModelNotAvailableError: If models are not loaded.
            KeyError: If a required feature is missing from feature_dict.
        """
        if self.classifier is None or self.regressor is None:
            raise ModelNotAvailableError("Model artifacts are not loaded.")

        ordered_values = [feature_dict[col] for col in FEATURE_COLUMNS]
        x = np.array(ordered_values, dtype=np.float64).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        level_encoded = self.classifier.predict(x_scaled)[0]
        level_probabilities = self.classifier.predict_proba(x_scaled)[0]
        confidence = float(np.max(level_probabilities))

        predicted_level = str(self.label_encoder.inverse_transform([level_encoded])[0])
        predicted_percentage = float(self.regressor.predict(x_scaled)[0])
        predicted_percentage = float(np.clip(predicted_percentage, 0.0, 100.0))

        return {
            "predicted_waste_level": predicted_level,
            "predicted_waste_percentage": round(predicted_percentage, 2),
            "confidence_score": round(confidence, 4),
        }


def get_predictor() -> WastePredictor:
    """FastAPI dependency-friendly accessor for the singleton predictor."""
    return WastePredictor.get_instance()
