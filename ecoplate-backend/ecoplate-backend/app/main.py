"""
EcoPlate AI backend application entrypoint.

Canteen-Scale Micro Food Wastage Quantification Using Plate-Level
Image Segmentation and Lightweight AI Inference.

Run locally with:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import API_DESCRIPTION, API_TITLE, API_VERSION
from app.database import init_db
from app.ml.predictor import ModelNotAvailableError, get_predictor
from app.routers import analytics, export, history, prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ecoplate.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup: initializes the database schema and warms up the ML
    model (training it if artifacts are not yet present) so the first
    incoming request does not pay the cold-start cost.
    """
    logger.info("Starting EcoPlate AI backend...")
    init_db()
    logger.info("Database initialized.")

    try:
        get_predictor()
        logger.info("ML model loaded and ready.")
    except ModelNotAvailableError:
        logger.exception("Model failed to load at startup.")

    yield
    logger.info("Shutting down EcoPlate AI backend.")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    contact={"name": "EcoPlate AI Team"},
    license_info={"name": "MIT"},
)

# CORS: open by default for prototype/demo purposes. Restrict allowed
# origins in production via environment-driven configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler ensuring uniform JSON error responses."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# Route registration
app.include_router(prediction.router)
app.include_router(history.router)
app.include_router(analytics.router)
app.include_router(export.router)


@app.get("/", tags=["Health"], summary="Root health check")
def read_root() -> dict:
    """Basic root endpoint confirming the service is running."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs_url": "/docs",
    }


@app.get("/api/health", tags=["Health"], summary="Detailed health check")
def health_check() -> dict:
    """
    Detailed health check verifying database and model availability.
    """
    model_status = "unavailable"
    try:
        get_predictor()
        model_status = "ready"
    except ModelNotAvailableError:
        model_status = "unavailable"

    return {
        "status": "ok",
        "database": "connected",
        "model": model_status,
    }
