"""
Feature extraction service.

Converts segmentation masks and the source image into a compact
numerical feature vector suitable for the Random Forest waste
predictor.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.segmentation import SegmentationResult


@dataclass
class ExtractedFeatures:
    """Numerical feature vector describing a plate's food/waste state."""

    plate_coverage_ratio: float
    leftover_area_px: float
    total_plate_area_px: float
    color_variance: float
    texture_score: float
    brightness_mean: float

    def to_array(self) -> np.ndarray:
        """Return features as an ordered numpy array for model input."""
        return np.array(
            [
                self.plate_coverage_ratio,
                self.leftover_area_px,
                self.total_plate_area_px,
                self.color_variance,
                self.texture_score,
                self.brightness_mean,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        """Return features as a plain dictionary."""
        return {
            "plate_coverage_ratio": self.plate_coverage_ratio,
            "leftover_area_px": self.leftover_area_px,
            "total_plate_area_px": self.total_plate_area_px,
            "color_variance": self.color_variance,
            "texture_score": self.texture_score,
            "brightness_mean": self.brightness_mean,
        }


def _compute_texture_score(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute a texture roughness score using the variance of the
    Laplacian restricted to the masked (food) region. Higher values
    indicate more textured / heterogeneous surfaces (typical of food),
    lower values indicate smooth, uniform surfaces (typical of an
    empty plate).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    masked_values = laplacian[mask > 0]
    if masked_values.size == 0:
        return 0.0
    return float(np.var(masked_values))


def _compute_color_variance(image: np.ndarray, mask: np.ndarray) -> float:
    """Compute the mean per-channel color variance within the masked region."""
    if np.count_nonzero(mask) == 0:
        return 0.0
    channel_variances = []
    for channel_index in range(3):
        channel = image[:, :, channel_index]
        values = channel[mask > 0]
        channel_variances.append(np.var(values))
    return float(np.mean(channel_variances))


def _compute_brightness_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean brightness (grayscale intensity) within the masked region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[mask > 0]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def extract_features(
    image: np.ndarray, segmentation: SegmentationResult
) -> ExtractedFeatures:
    """
    Extract a numerical feature vector from a preprocessed image and
    its corresponding segmentation result.

    Args:
        image: Preprocessed BGR image.
        segmentation: Output of the segmentation pipeline.

    Returns:
        ExtractedFeatures dataclass with all computed feature values.
    """
    plate_area = max(segmentation.plate_area_px, 1.0)
    coverage_ratio = float(
        np.clip(segmentation.food_area_px / plate_area, 0.0, 1.0)
    )

    color_variance = _compute_color_variance(image, segmentation.food_mask)
    texture_score = _compute_texture_score(image, segmentation.food_mask)
    brightness_mean = _compute_brightness_mean(image, segmentation.plate_mask)

    return ExtractedFeatures(
        plate_coverage_ratio=coverage_ratio,
        leftover_area_px=segmentation.leftover_area_px,
        total_plate_area_px=segmentation.plate_area_px,
        color_variance=color_variance,
        texture_score=texture_score,
        brightness_mean=brightness_mean,
    )
