"""
History API endpoints.

Allows persisting prediction results and retrieving a paginated,
filterable history of past predictions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.prediction import PredictionRecord
from app.schemas import HistoryListResponse, HistoryRecordResponse, SaveHistoryRequest

logger = logging.getLogger("ecoplate.history")

router = APIRouter(prefix="/api/history", tags=["History"])


@router.post(
    "",
    response_model=HistoryRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a prediction result to history",
)
def save_history(
    payload: SaveHistoryRequest, db: Session = Depends(get_db)
) -> HistoryRecordResponse:
    """Persist a prediction result into the database."""
    try:
        record = PredictionRecord(
            canteen_id=payload.canteen_id,
            meal_type=payload.meal_type,
            image_filename=payload.image_filename,
            food_category=payload.food_category,
            plate_coverage_ratio=payload.features.plate_coverage_ratio,
            leftover_area_px=payload.features.leftover_area_px,
            total_plate_area_px=payload.features.total_plate_area_px,
            color_variance=payload.features.color_variance,
            texture_score=payload.features.texture_score,
            brightness_mean=payload.features.brightness_mean,
            estimated_weight_grams=payload.estimated_weight_grams,
            predicted_waste_level=payload.predicted_waste_level,
            predicted_waste_percentage=payload.predicted_waste_percentage,
            confidence_score=payload.confidence_score,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return HistoryRecordResponse.model_validate(record)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Failed to save history record")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save history record: {exc}",
        ) from exc


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="Retrieve paginated prediction history",
)
def list_history(
    db: Session = Depends(get_db),
    canteen_id: Optional[str] = Query(default=None, description="Filter by canteen ID."),
    meal_type: Optional[str] = Query(default=None, description="Filter by meal type."),
    waste_level: Optional[str] = Query(default=None, description="Filter by waste level."),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=200, description="Records per page."),
) -> HistoryListResponse:
    """Retrieve a paginated, optionally filtered list of history records."""
    try:
        query = db.query(PredictionRecord)

        if canteen_id:
            query = query.filter(PredictionRecord.canteen_id == canteen_id)
        if meal_type:
            query = query.filter(PredictionRecord.meal_type == meal_type)
        if waste_level:
            query = query.filter(PredictionRecord.predicted_waste_level == waste_level)

        total = query.count()
        records = (
            query.order_by(PredictionRecord.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return HistoryListResponse(
            total=total,
            page=page,
            page_size=page_size,
            records=[HistoryRecordResponse.model_validate(r) for r in records],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {exc}",
        ) from exc


@router.get(
    "/{record_id}",
    response_model=HistoryRecordResponse,
    summary="Retrieve a single history record by ID",
)
def get_history_record(record_id: int, db: Session = Depends(get_db)) -> HistoryRecordResponse:
    """Fetch a single prediction history record by its primary key."""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with id {record_id} not found.",
        )
    return HistoryRecordResponse.model_validate(record)


@router.delete(
    "/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a history record by ID",
)
def delete_history_record(record_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a single prediction history record."""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History record with id {record_id} not found.",
        )
    try:
        db.delete(record)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Failed to delete history record")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete record: {exc}",
        ) from exc
