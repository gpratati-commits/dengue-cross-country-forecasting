"""
Country/setting-level robustness analysis for the dengue forecasting project.

Uses the strict common-cohort setting-level evaluation produced by
14_strict_common_evaluation.py.

The analysis examines:
- MAE, RMSE and MASE by setting, horizon and model
- performance across 1-, 2- and 4-week horizons
- how often MASE < 1
- variation in MASE across forecast horizons

Outputs:
    outputs/tables/country_robustness_detailed.csv
    outputs/tables/country_robustness_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TABLE_DIR = Path("outputs/tables")

INPUT_FILE = (
    TABLE_DIR
    / "strict_common_evaluation_by_setting.csv"
)

DETAILED_FILE = (
    TABLE_DIR
    / "country_robustness_detailed.csv"
)

SUMMARY_FILE = (
    TABLE_DIR
    / "country_robustness_summary.csv"
)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_HORIZONS = {
    1,
    2,
    4,
}

EXPECTED_MODELS = {
    "persistence",
    "seasonal_naive",
    "moving_average_4",
    "negative_binomial_epidemiology",
    "negative_binomial_climate",
    "xgboost_epidemiology",
    "xgboost_climate",
}


REQUIRED_COLUMNS = {
    "setting",
    "test_year",
    "horizon_weeks",
    "model",
    "n_predictions",
    "mae",
    "rmse",
    "mase",
}


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file: {INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


missing_columns = (
    REQUIRED_COLUMNS
    - set(df.columns)
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "test_year",
    "horizon_weeks",
    "n_predictions",
    "mae",
    "rmse",
    "mase",
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


if df[numeric_columns].isna().any().any():

    raise RuntimeError(
        "Missing or invalid numeric values "
        "were found in the strict evaluation."
    )


# ============================================================
# STRUCTURE CHECKS
# ============================================================

observed_horizons = set(
    df[
        "horizon_weeks"
    ]
    .astype(int)
    .unique()
)


if observed_horizons != EXPECTED_HORIZONS:

    raise RuntimeError(
        "Unexpected forecast horizons. "
        f"Observed: {sorted(observed_horizons)}"
    )


observed_models = set(
    df[
        "model"
    ]
    .astype(str)
    .unique()
)


if observed_models != EXPECTED_MODELS:

    missing_models = sorted(
        EXPECTED_MODELS
        - observed_models
    )

    extra_models = sorted(
        observed_models
        - EXPECTED_MODELS
    )

    raise RuntimeError(
        "Unexpected model coverage. "
        f"Missing models: {missing_models}. "
        f"Extra models: {extra_models}."
    )


duplicate_count = int(
    df.duplicated(
        subset=[
            "setting",
            "test_year",
            "horizon_weeks",
            "model",
        ]
    ).sum()
)


if duplicate_count != 0:

    raise RuntimeError(
        f"Duplicate evaluation rows found: "
        f"{duplicate_count}"
    )


# ============================================================
# ADD ROBUSTNESS INDICATORS
# ============================================================

df[
    "mase_below_1"
] = (
    df[
        "mase"
    ]
    <
    1
)


df[
    "mase_equal_or_above_1"
] = (
    df[
        "mase"
    ]
    >=
    1
)


df = df.sort_values(
    [
        "setting",
        "horizon_weeks",
        "model",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE DETAILED TABLE
# ============================================================

df.to_csv(
    DETAILED_FILE,
    index=False,
)


# ============================================================
# CROSS-HORIZON SUMMARY
# ============================================================

summary = (
    df
    .groupby(
        [
            "setting",
            "model",
        ],
        as_index=False,
    )
    .agg(
        horizons_evaluated=(
            "horizon_weeks",
            "nunique",
        ),
        total_predictions=(
            "n_predictions",
            "sum",
        ),
        median_mae=(
            "mae",
            "median",
        ),
        median_rmse=(
            "rmse",
            "median",
        ),
        median_mase=(
            "mase",
            "median",
        ),
        min_mase=(
            "mase",
            "min",
        ),
        max_mase=(
            "mase",
            "max",
        ),
        horizons_mase_below_1=(
            "mase_below_1",
            "sum",
        ),
    )
)


summary[
    "mase_range"
] = (
    summary[
        "max_mase"
    ]
    -
    summary[
        "min_mase"
    ]
)


summary[
    "fraction_horizons_mase_below_1"
] = (
    summary[
        "horizons_mase_below_1"
    ]
    /
    summary[
        "horizons_evaluated"
    ]
)


summary[
    "all_horizons_mase_below_1"
] = (
    summary[
        "horizons_mase_below_1"
    ]
    ==
    summary[
        "horizons_evaluated"
    ]
)


summary = summary.sort_values(
    [
        "setting",
        "median_mase",
        "model",
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if summary.empty:

    raise RuntimeError(
        "Country robustness summary is empty."
    )


if (
    summary[
        "horizons_evaluated"
    ]
    !=
    3
).any():

    raise RuntimeError(
        "One or more setting/model combinations "
        "do not contain all three horizons."
    )


if not np.isfinite(
    summary[
        [
            "median_mae",
            "median_rmse",
            "median_mase",
            "min_mase",
            "max_mase",
            "mase_range",
        ]
    ].to_numpy(
        dtype=float
    )
).all():

    raise RuntimeError(
        "Non-finite robustness statistics found."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# PRINT DETAILED RESULTS BY SETTING
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "COUNTRY / SETTING ROBUSTNESS ANALYSIS"
)

print(
    "=" * 80
)


settings = sorted(
    summary[
        "setting"
    ].unique()
)


for setting in settings:

    setting_summary = summary[
        summary[
            "setting"
        ]
        ==
        setting
    ].copy()


    print(
        "\n"
        + "-" * 80
    )

    print(
        setting.upper()
    )

    print(
        "-" * 80
    )


    print(
        setting_summary[
            [
                "model",
                "median_mae",
                "median_rmse",
                "median_mase",
                "min_mase",
                "max_mase",
                "mase_range",
                "horizons_mase_below_1",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# OVERALL COVERAGE
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "ROBUSTNESS COVERAGE"
)

print(
    "=" * 80
)


print(
    "Settings:",
    sorted(
        df[
            "setting"
        ].unique()
    ),
)


print(
    "Horizons:",
    sorted(
        df[
            "horizon_weeks"
        ]
        .astype(int)
        .unique()
    ),
)


print(
    "Models:",
    sorted(
        df[
            "model"
        ].unique()
    ),
)


print(
    "Detailed rows:",
    len(df),
)


print(
    "Summary rows:",
    len(summary),
)


print(
    "\nCOUNTRY ROBUSTNESS ANALYSIS COMPLETE"
)


print(
    f"Saved: {DETAILED_FILE}"
)


print(
    f"Saved: {SUMMARY_FILE}"
)