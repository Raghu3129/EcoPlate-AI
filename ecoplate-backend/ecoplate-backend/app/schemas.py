"""
Pydantic schemas used for request validation and response
serialization across the EcoPlate AI API.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class FeatureVector(BaseModel):
    """Numerical features extracted from a plate image."""

    plate_coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    leftover_area_px: float = Field(..., ge=0.0)
    total_plate_area_px: float = Field(..., ge=0.0)
    color_variance: float = Field(..., ge=0.0)
    texture_score: float = Field(..., ge=0.0)
    brightness_mean: float = Field(..., ge=0.0, le=255.0)


class PredictionRequest(BaseModel):
    """Optional metadata accompanying an image prediction request."""

    canteen_id: str = Field(default="default", max_length=100)
    meal_type: str = Field(default="unspecified", max_length=50)
    food_category_hint: Optional[str] = Field(default=None, max_length=50)


class PredictionResponse(BaseModel):
    """Response returned after running inference on a plate image."""

    food_category: str
    features: FeatureVector
    predicted_waste_level: str
    predicted_waste_percentage: float
    estimated_weight_grams: float
    confidence_score: float

    model_config = ConfigDict(from_attributes=True)


class SaveHistoryRequest(BaseModel):
    """Payload for persisting a prediction into history."""

    canteen_id: str = Field(default="default", max_length=100)
    meal_type: str = Field(default="unspecified", max_length=50)
    image_filename: Optional[str] = None
    food_category: str
    features: FeatureVector
    predicted_waste_level: str
    predicted_waste_percentage: float
    estimated_weight_grams: float
    confidence_score: float


class HistoryRecordResponse(BaseModel):
    """A single historical prediction record."""

    id: int
    canteen_id: str
    meal_type: str
    image_filename: Optional[str]
    food_category: str
    plate_coverage_ratio: float
    leftover_area_px: float
    total_plate_area_px: float
    color_variance: float
    texture_score: float
    brightness_mean: float
    estimated_weight_grams: float
    predicted_waste_level: str
    predicted_waste_percentage: float
    confidence_score: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HistoryListResponse(BaseModel):
    """Paginated list of history records."""

    total: int
    page: int
    page_size: int
    records: List[HistoryRecordResponse]


class AnalyticsSummary(BaseModel):
    """Aggregate analytics computed over stored prediction history."""

    total_predictions: int
    average_waste_percentage: float
    average_estimated_weight_grams: float
    waste_level_distribution: dict
    food_category_distribution: dict
    average_confidence_score: float
    total_estimated_waste_grams: float


class ErrorResponse(BaseModel):
    """Standard error envelope returned on failures."""

    error: str
    detail: Optional[str] = None
