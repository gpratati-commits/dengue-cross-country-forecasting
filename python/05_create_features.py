"""
Create leakage-safe features for multi-setting dengue forecasting.

Features include:
- recent dengue case history
- lagged ERA5-Land climate variables
- rolling recent dengue activity
- seasonal Fourier terms
- 1-, 2-, and 4-week forecast targets

Missing surveillance periods remain missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/dengue_weekly_clean.csv"
)

OUTPUT_FILE = Path(
    "data/processed/dengue_model_features.csv"
)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

df = (
    df
    .sort_values(
        ["setting", "analysis_week"]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Transformations
# ------------------------------------------------------------

df["log_cases"] = np.log1p(
    df["cases"]
)

df["log_rain_era5"] = np.log1p(
    df["rain_era5"].clip(lower=0)
)


CASE_LAGS = [
    1,
    2,
    4,
    8,
    52
]

CLIMATE_LAGS = [
    0,
    1,
    2,
    4,
    6,
    8
]


def create_setting_features(group):

    group = (
        group
        .sort_values("analysis_week")
        .copy()
    )

    # ----------------------------------------
    # Dengue history
    # ----------------------------------------

    for lag in CASE_LAGS:

        group[
            f"log_cases_lag_{lag}"
        ] = (
            group["log_cases"]
            .shift(lag)
        )

    # ----------------------------------------
    # Climate history
    # ----------------------------------------

    climate_variables = [
        "log_rain_era5",
        "temp_era5",
        "humidity_era5"
    ]

    for variable in climate_variables:

        for lag in CLIMATE_LAGS:

            group[
                f"{variable}_lag_{lag}"
            ] = (
                group[variable]
                .shift(lag)
            )

    # ----------------------------------------
    # Recent dengue activity
    # Uses ONLY previous weeks
    # ----------------------------------------

    group["log_cases_roll4"] = (
        group["log_cases"]
        .shift(1)
        .rolling(
            window=4,
            min_periods=4
        )
        .mean()
    )

    # ----------------------------------------
    # Seasonality
    # ----------------------------------------

    week = (
        group["source_week"]
        .astype(float)
    )

    group["season_sin"] = np.sin(
        2 * np.pi * week / 52
    )

    group["season_cos"] = np.cos(
        2 * np.pi * week / 52
    )

    # ----------------------------------------
    # Forecast targets
    # ----------------------------------------

    for horizon in [
        1,
        2,
        4
    ]:

        group[
            f"target_log_h{horizon}"
        ] = (
            group["log_cases"]
            .shift(-horizon)
        )

        group[
            f"target_cases_h{horizon}"
        ] = (
            group["cases"]
            .shift(-horizon)
        )

    return group


feature_parts = []

for _, group in df.groupby(
    "setting",
    sort=False
):

    feature_parts.append(
        create_setting_features(
            group
        )
    )


features = pd.concat(
    feature_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

assert not (
    features.loc[
        features["structural_gap"],
        "cases"
    ]
    .notna()
    .any()
)

print(
    "Structural-gap safety check: PASS"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

features.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nFEATURE ENGINEERING COMPLETE"
)

print(
    f"Rows: {len(features):,}"
)

print(
    f"Columns: {features.shape[1]}"
)

print(
    f"Saved: {OUTPUT_FILE}"
)