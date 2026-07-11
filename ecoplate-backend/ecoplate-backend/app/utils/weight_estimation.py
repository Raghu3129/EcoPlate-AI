"""
Utility for converting pixel-area leftover measurements into an
estimated physical weight in grams.

Uses category-specific density heuristics (grams per leftover pixel)
consistent with those used in synthetic data generation, so trained
model outputs and heuristic weight estimates remain aligned.
"""

from app.ml.synthetic_data_generator import CATEGORY_PROFILES


DEFAULT_DENSITY_G_PER_PX = 0.045


def estimate_leftover_weight_grams(
    leftover_area_px: float, food_category: str
) -> float:
    """
    Estimate the weight (in grams) of leftover food on a plate.

    Args:
        leftover_area_px: Number of pixels classified as leftover /
            uneaten food or residue.
        food_category: One of the known food category labels; falls
            back to a default density if unrecognized.

    Returns:
        Estimated weight in grams, rounded to two decimal places.
    """
    profile = CATEGORY_PROFILES.get(food_category)
    density = profile["density_g_per_px"] if profile else DEFAULT_DENSITY_G_PER_PX
    weight = max(0.0, leftover_area_px * density)
    return round(weight, 2)
