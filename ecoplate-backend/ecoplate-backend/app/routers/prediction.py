"""
Prediction API endpoints.

Accepts a plate image upload, runs the full preprocessing ->
segmentation -> feature extraction -> Random Forest inference
pipeline, and returns the predicted waste level, percentage, and
estimated leftover weight.
"""
from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.prediction import PredictionRecord
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.ml.predictor import ModelNotAvailableError, get_predictor
from app.schemas import FeatureVector, PredictionResponse
from app.services.feature_extraction import extract_features
from app.services.image_processing import ImageProcessingError, preprocess_pipeline
from app.services.segmentation import segment_plate
from app.utils.weight_estimation import estimate_leftover_weight_grams

logger = logging.getLogger("ecoplate.prediction")

router = APIRouter(prefix="/api/predict", tags=["Prediction"])


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    """Validate file extension and size before processing."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{suffix}'. Allowed: "
                f"{sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            ),
        )

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict food waste from a plate image",
    description=(
        "Upload a canteen plate image to receive a predicted waste "
        "level, waste percentage, and estimated leftover weight in "
        "grams, derived via OpenCV segmentation and a Random Forest "
        "model."
    ),
)
async def predict_waste(
    file: UploadFile = File(...),
    food_category_hint: str = Form(default="mixed"),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """
    Run the full inference pipeline on an uploaded plate image.

    Raises:
        HTTPException 400/415/413: On invalid input.
        HTTPException 500: On internal processing or model failure.
    """
    try:
        contents = await file.read()
        _validate_upload(file, contents)

        # --- Preprocessing ---
        try:
            preprocessed_image = preprocess_pipeline(contents)
        except ImageProcessingError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        # --- Segmentation ---
        segmentation_result = segment_plate(preprocessed_image)

        # --- Feature extraction ---
        features = extract_features(preprocessed_image, segmentation_result)
        feature_dict = features.to_dict()

        # --- Weight estimation (heuristic, category-aware) ---
        estimated_weight = estimate_leftover_weight_grams(
            features.leftover_area_px, food_category_hint
        )

        # --- Model inference ---
        try:
            predictor = get_predictor()
            prediction = predictor.predict(feature_dict)
        except ModelNotAvailableError as exc:
            logger.exception("Model unavailable during prediction")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Prediction model unavailable: {exc}",
            ) from exc

        # ----------------------------
        # Save prediction to database
        # ----------------------------

        record = PredictionRecord(
            canteen_id="Default Canteen",
            meal_type="Lunch",
            image_filename=file.filename,

            food_category=food_category_hint,

            plate_coverage_ratio=features.plate_coverage_ratio,
            leftover_area_px=features.leftover_area_px,
            total_plate_area_px=features.total_plate_area_px,
            color_variance=features.color_variance,
            texture_score=features.texture_score,
            brightness_mean=features.brightness_mean,

            estimated_weight_grams=estimated_weight,

            predicted_waste_level=prediction["predicted_waste_level"],
            predicted_waste_percentage=prediction["predicted_waste_percentage"],
            confidence_score=prediction["confidence_score"],
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return PredictionResponse(
            food_category=food_category_hint,
            features=FeatureVector(**feature_dict),
            predicted_waste_level=prediction["predicted_waste_level"],
            predicted_waste_percentage=prediction["predicted_waste_percentage"],
            estimated_weight_grams=estimated_weight,
            confidence_score=prediction["confidence_score"],
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during prediction: {exc}",
        ) from exc
