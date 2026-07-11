"""
Synthetic dataset generator for training the Random Forest waste
predictor.

Since collecting thousands of real canteen plate images with verified
ground-truth waste weight is impractical for a prototype, this module
generates statistically realistic synthetic feature vectors that mimic
the distributions expected from real segmentation output, conditioned
on food category and waste level.

The generative process models plausible correlations, e.g.:
- Higher leftover area correlates with higher waste percentage.
- Lower plate coverage ratio correlates with higher waste levels.
- Texture/color variance differs by food category (rice vs curry vs
  salad have different visual heterogeneity).
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import (
    FOOD_CATEGORIES,
    RANDOM_SEED,
    SYNTHETIC_DATASET_PATH,
    SYNTHETIC_SAMPLE_COUNT,
    WASTE_LEVELS,
)

# Waste level -> (min, max) waste percentage range.
WASTE_PERCENTAGE_RANGES: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 15.0),
    "medium": (15.0, 40.0),
    "high": (40.0, 70.0),
    "severe": (70.0, 100.0),
}

# Food category -> base texture/color heterogeneity multipliers.
# Values are illustrative priors used only for synthetic generation.
CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "rice": {"texture_base": 120.0, "color_var_base": 90.0, "density_g_per_px": 0.045},
    "curry": {"texture_base": 220.0, "color_var_base": 260.0, "density_g_per_px": 0.05},
    "vegetables": {"texture_base": 300.0, "color_var_base": 320.0, "density_g_per_px": 0.04},
    "bread": {"texture_base": 90.0, "color_var_base": 60.0, "density_g_per_px": 0.03},
    "dal": {"texture_base": 60.0, "color_var_base": 70.0, "density_g_per_px": 0.048},
    "salad": {"texture_base": 350.0, "color_var_base": 400.0, "density_g_per_px": 0.035},
    "dessert": {"texture_base": 100.0, "color_var_base": 150.0, "density_g_per_px": 0.055},
    "mixed": {"texture_base": 250.0, "color_var_base": 280.0, "density_g_per_px": 0.045},
}


def _sample_waste_level(rng: np.random.Generator) -> str:
    """Sample a waste level with a realistic non-uniform prior."""
    # Canteens typically see more low/medium waste than severe waste.
    probabilities = [0.35, 0.32, 0.22, 0.11]
    return str(rng.choice(WASTE_LEVELS, p=probabilities))


def _generate_single_sample(rng: np.random.Generator) -> dict:
    """Generate a single synthetic feature row with a plausible label."""
    food_category = str(rng.choice(FOOD_CATEGORIES))
    waste_level = _sample_waste_level(rng)
    profile = CATEGORY_PROFILES[food_category]

    low, high = WASTE_PERCENTAGE_RANGES[waste_level]
    waste_percentage = float(rng.uniform(low, high))

    # Total plate area varies slightly per image capture distance.
    total_plate_area_px = float(rng.normal(45000, 4000))
    total_plate_area_px = max(total_plate_area_px, 20000.0)

    # Plate coverage ratio inversely related to waste percentage, with noise.
    base_coverage = 1.0 - (waste_percentage / 100.0)
    plate_coverage_ratio = float(np.clip(rng.normal(base_coverage, 0.05), 0.0, 1.0))

    # Leftover area derived from waste percentage and total plate area.
    leftover_area_px = float(
        np.clip((waste_percentage / 100.0) * total_plate_area_px * rng.uniform(0.85, 1.15), 0, total_plate_area_px)
    )

    # Color variance and texture scaled by category profile and coverage.
    color_variance = float(
        max(0.0, rng.normal(profile["color_var_base"] * (0.5 + plate_coverage_ratio), 25))
    )
    texture_score = float(
        max(0.0, rng.normal(profile["texture_base"] * (0.5 + plate_coverage_ratio), 20))
    )

    # Brightness mean: emptier plates (more visible ceramic/steel) tend brighter.
    brightness_mean = float(
        np.clip(rng.normal(140 - (plate_coverage_ratio * 20), 15), 40, 240)
    )

    # Estimated remaining food weight from leftover area and category density.
    # Higher waste_percentage -> more food left as "leftover" relative to
    # what was originally served; here we model estimated *wasted* weight.
    estimated_weight_grams = float(
        max(0.0, leftover_area_px * profile["density_g_per_px"] * rng.uniform(0.9, 1.1))
    )

    return {
        "food_category": food_category,
        "plate_coverage_ratio": round(plate_coverage_ratio, 5),
        "leftover_area_px": round(leftover_area_px, 2),
        "total_plate_area_px": round(total_plate_area_px, 2),
        "color_variance": round(color_variance, 4),
        "texture_score": round(texture_score, 4),
        "brightness_mean": round(brightness_mean, 2),
        "estimated_weight_grams": round(estimated_weight_grams, 2),
        "waste_percentage": round(waste_percentage, 2),
        "waste_level": waste_level,
    }


def generate_synthetic_dataset(
    n_samples: int = SYNTHETIC_SAMPLE_COUNT, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of plate feature vectors and waste
    labels.

    Args:
        n_samples: Number of synthetic samples to generate (>= 3000
            recommended for stable Random Forest training).
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with feature columns, waste_percentage and
        waste_level target columns.
    """
    rng = np.random.default_rng(seed)
    rows: List[dict] = [_generate_single_sample(rng) for _ in range(n_samples)]
    return pd.DataFrame(rows)


def save_synthetic_dataset(df: pd.DataFrame) -> str:
    """Persist the generated dataset to CSV and return the file path."""
    df.to_csv(SYNTHETIC_DATASET_PATH, index=False)
    return str(SYNTHETIC_DATASET_PATH)


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    path = save_synthetic_dataset(dataset)
    print(f"Generated {len(dataset)} synthetic samples -> {path}")
    print(dataset["waste_level"].value_counts())
