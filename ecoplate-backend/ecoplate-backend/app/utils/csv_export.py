"""CSV export utility for prediction history records."""

import csv
import io
from typing import List

from app.models.prediction import PredictionRecord

CSV_COLUMNS = [
    "id",
    "canteen_id",
    "meal_type",
    "image_filename",
    "food_category",
    "plate_coverage_ratio",
    "leftover_area_px",
    "total_plate_area_px",
    "color_variance",
    "texture_score",
    "brightness_mean",
    "estimated_weight_grams",
    "predicted_waste_level",
    "predicted_waste_percentage",
    "confidence_score",
    "created_at",
]


def records_to_csv(records: List[PredictionRecord]) -> str:
    """
    Convert a list of PredictionRecord ORM objects into an in-memory
    CSV string.

    Args:
        records: List of PredictionRecord instances.

    Returns:
        A string containing CSV-formatted data with a header row.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for record in records:
        row = record.to_dict()
        writer.writerow(row)

    return buffer.getvalue()
