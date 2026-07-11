"""
Export API endpoints.

Provides CSV export of prediction history records for offline
analysis or reporting.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.prediction import PredictionRecord
from app.utils.csv_export import records_to_csv

logger = logging.getLogger("ecoplate.export")

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.get(
    "/csv",
    summary="Export prediction history as a CSV file",
    response_description="A downloadable CSV file of prediction history.",
)
def export_history_csv(
    db: Session = Depends(get_db),
    canteen_id: Optional[str] = Query(default=None, description="Filter by canteen ID."),
    meal_type: Optional[str] = Query(default=None, description="Filter by meal type."),
) -> StreamingResponse:
    """Stream the filtered prediction history as a downloadable CSV file."""
    try:
        query = db.query(PredictionRecord)
        if canteen_id:
            query = query.filter(PredictionRecord.canteen_id == canteen_id)
        if meal_type:
            query = query.filter(PredictionRecord.meal_type == meal_type)

        records = query.order_by(PredictionRecord.created_at.desc()).all()
        csv_content = records_to_csv(records)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"ecoplate_history_{timestamp}.csv"

        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to export CSV")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export CSV: {exc}",
        ) from exc
