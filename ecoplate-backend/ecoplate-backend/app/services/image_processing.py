"""
Image preprocessing utilities built on OpenCV.

Responsible for decoding raw image bytes, normalizing size, denoising,
and preparing images for downstream segmentation.
"""

from typing import Tuple

import cv2
import numpy as np

from app.config import IMAGE_RESIZE_DIM


class ImageProcessingError(Exception):
    """Raised when an uploaded image cannot be decoded or processed."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR OpenCV image array.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        A numpy ndarray representing the decoded BGR image.

    Raises:
        ImageProcessingError: If the bytes cannot be decoded as an image.
    """
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError(
            "Unable to decode image. File may be corrupted or in an "
            "unsupported format."
        )
    return image


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int] = IMAGE_RESIZE_DIM
) -> np.ndarray:
    """Resize an image to a fixed target size using area interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply a light bilateral filter to reduce noise while keeping edges."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE on the luminance channel.

    Improves segmentation robustness under uneven canteen lighting.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(image_bytes: bytes) -> np.ndarray:
    """
    Run the full preprocessing pipeline on raw uploaded image bytes.

    Steps: decode -> resize -> denoise -> contrast enhancement.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        A preprocessed BGR image ready for segmentation.
    """
    image = decode_image(image_bytes)
    image = resize_image(image)
    image = denoise_image(image)
    image = enhance_contrast(image)
    return image
