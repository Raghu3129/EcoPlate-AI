"""
SQLAlchemy ORM model representing a single plate-waste prediction
record stored in the history table.
"""

from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    """Return a timezone-aware current UTC timestamp."""
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """
    Persisted record of a food-waste prediction made on a canteen
    plate image, including extracted features and model output.
    """

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Metadata
    canteen_id: Mapped[str] = mapped_column(String(100), default="default", nullable=False)
    meal_type: Mapped[str] = mapped_column(String(50), default="unspecified", nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=True)

    # Extracted plate / food features
    food_category: Mapped[str] = mapped_column(String(50), nullable=False)
    plate_coverage_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    leftover_area_px: Mapped[float] = mapped_column(Float, nullable=False)
    total_plate_area_px: Mapped[float] = mapped_column(Float, nullable=False)
    color_variance: Mapped[float] = mapped_column(Float, nullable=False)
    texture_score: Mapped[float] = mapped_column(Float, nullable=False)
    brightness_mean: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_weight_grams: Mapped[float] = mapped_column(Float, nullable=False)

    # Model output
    predicted_waste_level: Mapped[str] = mapped_column(String(20), nullable=False)
    predicted_waste_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Serialize the ORM record into a JSON-friendly dictionary."""
        return {
            "id": self.id,
            "canteen_id": self.canteen_id,
            "meal_type": self.meal_type,
            "image_filename": self.image_filename,
            "food_category": self.food_category,
            "plate_coverage_ratio": round(self.plate_coverage_ratio, 4),
            "leftover_area_px": round(self.leftover_area_px, 2),
            "total_plate_area_px": round(self.total_plate_area_px, 2),
            "color_variance": round(self.color_variance, 4),
            "texture_score": round(self.texture_score, 4),
            "brightness_mean": round(self.brightness_mean, 2),
            "estimated_weight_grams": round(self.estimated_weight_grams, 2),
            "predicted_waste_level": self.predicted_waste_level,
            "predicted_waste_percentage": round(self.predicted_waste_percentage, 2),
            "confidence_score": round(self.confidence_score, 4),
            "created_at": self.created_at.isoformat(),
        }
