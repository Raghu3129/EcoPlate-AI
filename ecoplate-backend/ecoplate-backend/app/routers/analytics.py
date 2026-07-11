"""
Analytics API endpoints.

Provides aggregate statistics computed over the stored prediction
history, useful for canteen-level dashboards.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.prediction import PredictionRecord
from app.schemas import AnalyticsSummary

logger = logging.getLogger("ecoplate.analytics")

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    response_model=AnalyticsSummary,
    summary="Get aggregate waste analytics summary",
)
def get_analytics_summary(
    db: Session = Depends(get_db),
    canteen_id: Optional[str] = Query(default=None, description="Filter by canteen ID."),
) -> AnalyticsSummary:
    """
    Compute aggregate analytics (averages and distributions) across
    all stored prediction history, optionally scoped to one canteen.
    """
    try:
        query = db.query(PredictionRecord)
        if canteen_id:
            query = query.filter(PredictionRecord.canteen_id == canteen_id)

        total_predictions = query.count()

        if total_predictions == 0:
            return AnalyticsSummary(
                total_predictions=0,
                average_waste_percentage=0.0,
                average_estimated_weight_grams=0.0,
                waste_level_distribution={},
                food_category_distribution={},
                average_confidence_score=0.0,
                total_estimated_waste_grams=0.0,
            )

        avg_waste_percentage = query.with_entities(
            func.avg(PredictionRecord.predicted_waste_percentage)
        ).scalar() or 0.0

        avg_weight = query.with_entities(
            func.avg(PredictionRecord.estimated_weight_grams)
        ).scalar() or 0.0

        avg_confidence = query.with_entities(
            func.avg(PredictionRecord.confidence_score)
        ).scalar() or 0.0

        total_weight = query.with_entities(
            func.sum(PredictionRecord.estimated_weight_grams)
        ).scalar() or 0.0

        level_rows = (
            query.with_entities(
                PredictionRecord.predicted_waste_level, func.count(PredictionRecord.id)
            )
            .group_by(PredictionRecord.predicted_waste_level)
            .all()
        )
        waste_level_distribution = {level: count for level, count in level_rows}

        category_rows = (
            query.with_entities(
                PredictionRecord.food_category, func.count(PredictionRecord.id)
            )
            .group_by(PredictionRecord.food_category)
            .all()
        )
        food_category_distribution = {category: count for category, count in category_rows}

        return AnalyticsSummary(
            total_predictions=total_predictions,
            average_waste_percentage=round(float(avg_waste_percentage), 2),
            average_estimated_weight_grams=round(float(avg_weight), 2),
            waste_level_distribution=waste_level_distribution,
            food_category_distribution=food_category_distribution,
            average_confidence_score=round(float(avg_confidence), 4),
            total_estimated_waste_grams=round(float(total_weight), 2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to compute analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute analytics: {exc}",
        ) from exc


@router.get(
    "/trend",
    summary="Get daily average waste percentage trend",
)
def get_waste_trend(
    db: Session = Depends(get_db),
    canteen_id: Optional[str] = Query(default=None, description="Filter by canteen ID."),
    limit_days: int = Query(default=30, ge=1, le=365, description="Number of most recent days to include."),
) -> dict:
    """Return a day-wise average waste percentage trend for charting."""
    try:
        query = db.query(
            func.date(PredictionRecord.created_at).label("day"),
            func.avg(PredictionRecord.predicted_waste_percentage).label("avg_waste"),
            func.count(PredictionRecord.id).label("count"),
        )
        if canteen_id:
            query = query.filter(PredictionRecord.canteen_id == canteen_id)

        rows = (
            query.group_by("day")
            .order_by(func.date(PredictionRecord.created_at).desc())
            .limit(limit_days)
            .all()
        )

        trend = [
            {"date": str(row.day), "average_waste_percentage": round(float(row.avg_waste), 2), "count": row.count}
            for row in rows
        ]
        trend.reverse()
        return {"trend": trend}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to compute waste trend")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute waste trend: {exc}",
        ) from exc
