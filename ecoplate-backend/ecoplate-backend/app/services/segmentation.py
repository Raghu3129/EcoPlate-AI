"""
Plate and food-region segmentation using classical OpenCV techniques.

This module implements a lightweight segmentation pipeline suitable for
CPU-only, low-resource canteen deployments:

1. Detect the circular plate region (Hough circle transform, with a
   contour-based ellipse fallback for non-ideal camera angles).
2. Within the plate mask, separate "food" pixels from "empty plate /
   leftover residue" pixels using HSV color thresholding + Otsu
   thresholding on saturation.
3. Return binary masks for plate, food, and leftover regions.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Container for masks and areas produced by segmentation."""

    plate_mask: np.ndarray
    food_mask: np.ndarray
    leftover_mask: np.ndarray
    plate_area_px: float
    food_area_px: float
    leftover_area_px: float


def _detect_plate_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect the plate region and return a binary mask.

    Attempts a Hough circle transform first (works well for top-down
    canteen tray shots); falls back to the largest contour on an
    adaptive threshold if no circle is confidently found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    height, width = gray.shape[:2]

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=height // 2,
        param1=100,
        param2=40,
        minRadius=int(min(height, width) * 0.25),
        maxRadius=int(min(height, width) * 0.5),
    )

    mask = np.zeros((height, width), dtype=np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(mask, (int(x), int(y)), int(r), 255, thickness=-1)
        return mask

    # Fallback: largest contour from adaptive threshold.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    else:
        # Last resort: assume a centered circular plate covering 70% of frame.
        center = (width // 2, height // 2)
        radius = int(min(height, width) * 0.35)
        cv2.circle(mask, center, radius, 255, thickness=-1)

    return mask


def _detect_food_vs_leftover(
    image: np.ndarray, plate_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Within the plate region, separate food pixels from empty/leftover
    (bare plate, sauce smears, scraps) pixels using HSV saturation and
    Otsu's method.

    Food items generally exhibit higher color saturation and variance
    than a mostly-empty ceramic/steel plate surface.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Restrict Otsu computation to plate pixels only.
    masked_saturation = cv2.bitwise_and(saturation, saturation, mask=plate_mask)

    plate_pixel_count = int(np.count_nonzero(plate_mask))
    if plate_pixel_count == 0:
        empty = np.zeros_like(plate_mask)
        return empty, empty

    _, food_mask = cv2.threshold(
        masked_saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    food_mask = cv2.bitwise_and(food_mask, food_mask, mask=plate_mask)

    # Morphological cleanup to remove speckle noise.
    kernel = np.ones((5, 5), np.uint8)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel)

    leftover_mask = cv2.bitwise_and(
        plate_mask, cv2.bitwise_not(food_mask)
    )

    return food_mask, leftover_mask


def segment_plate(image: np.ndarray) -> SegmentationResult:
    """
    Run the full segmentation pipeline on a preprocessed image.

    Args:
        image: A preprocessed BGR image (see image_processing module).

    Returns:
        SegmentationResult containing binary masks and pixel areas for
        the plate, food, and leftover/empty regions.
    """
    plate_mask = _detect_plate_mask(image)
    food_mask, leftover_mask = _detect_food_vs_leftover(image, plate_mask)

    plate_area = float(np.count_nonzero(plate_mask))
    food_area = float(np.count_nonzero(food_mask))
    leftover_area = float(np.count_nonzero(leftover_mask))

    return SegmentationResult(
        plate_mask=plate_mask,
        food_mask=food_mask,
        leftover_mask=leftover_mask,
        plate_area_px=plate_area,
        food_area_px=food_area,
        leftover_area_px=leftover_area,
    )
