"""
Climate contribution analysis for the dengue forecasting project.

This script compares climate-informed models with their corresponding
epidemiology-only models on the strict common forecast cohort.

Paired comparisons:

1. Negative binomial:
   negative_binomial_epidemiology
   vs
   negative_binomial_climate

2. XGBoost:
   xgboost_epidemiology
   vs
   xgboost_climate

Delta convention:

    delta = climate model error - epidemiology-only model error

Therefore:

    delta < 0  -> climate model has lower error
    delta > 0  -> climate model has higher error
    delta = 0  -> equal error

Outputs:

    outputs/tables/climate_contribution_by_setting.csv
    outputs/tables/climate_contribution_summary.csv
    outputs/predictions/climate_contribution_paired_targets.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TABLE_DIR = Path("outputs/tables")
PREDICTION_DIR = Path("outputs/predictions")

SETTING_EVALUATION_FILE = (
    TABLE_DIR
    / "strict_common_evaluation_by_setting.csv"
)

COMMON_PREDICTION_FILE = (
    PREDICTION_DIR
    / "strict_common_cohort_predictions.csv"
)


TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL PAIRS
# ============================================================

MODEL_PAIRS = {
    "negative_binomial": {
        "epidemiology":
            "negative_binomial_epidemiology",
        "climate":
            "negative_binomial_climate",
    },
    "xgboost": {
        "epidemiology":
            "xgboost_epidemiology",
        "climate":
            "xgboost_climate",
    },
}


EXPECTED_HORIZONS = {
    1,
    2,
    4,
}


# ============================================================
# CHECK INPUT FILES
# ============================================================

for path in [
    SETTING_EVALUATION_FILE,
    COMMON_PREDICTION_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input file: {path}"
        )


# ============================================================
# READ SETTING-LEVEL EVALUATION
# ============================================================

evaluation = pd.read_csv(
    SETTING_EVALUATION_FILE
)


required_evaluation_columns = {
    "setting",
    "test_year",
    "horizon_weeks",
    "model",
    "n_predictions",
    "mae",
    "rmse",
    "mase",
}


missing_columns = (
    required_evaluation_columns
    - set(evaluation.columns)
)


if missing_columns:

    raise ValueError(
        "Missing evaluation columns: "
        + ", ".join(
            sorted(missing_columns)
        )
    )


numeric_columns = [
    "test_year",
    "horizon_weeks",
    "n_predictions",
    "mae",
    "rmse",
    "mase",
]


for column in numeric_columns:

    evaluation[column] = pd.to_numeric(
        evaluation[column],
        errors="coerce",
    )


# ============================================================
# CHECK HORIZONS
# ============================================================

observed_horizons = set(
    evaluation[
        "horizon_weeks"
    ]
    .dropna()
    .astype(int)
    .unique()
)


if observed_horizons != EXPECTED_HORIZONS:

    raise RuntimeError(
        "Unexpected forecast horizons. "
        f"Expected {sorted(EXPECTED_HORIZONS)}, "
        f"observed {sorted(observed_horizons)}."
    )


# ============================================================
# SETTING-LEVEL PAIRED COMPARISON
# ============================================================

comparison_frames = []


for family, pair in MODEL_PAIRS.items():

    epidemiology_model = pair[
        "epidemiology"
    ]

    climate_model = pair[
        "climate"
    ]


    epidemiology = evaluation[
        evaluation[
            "model"
        ]
        ==
        epidemiology_model
    ].copy()


    climate = evaluation[
        evaluation[
            "model"
        ]
        ==
        climate_model
    ].copy()


    if epidemiology.empty:

        raise RuntimeError(
            f"No rows found for {epidemiology_model}."
        )


    if climate.empty:

        raise RuntimeError(
            f"No rows found for {climate_model}."
        )


    epidemiology = epidemiology.rename(
        columns={
            "n_predictions":
                "n_predictions_epidemiology",
            "mae":
                "mae_epidemiology",
            "rmse":
                "rmse_epidemiology",
            "mase":
                "mase_epidemiology",
        }
    )


    climate = climate.rename(
        columns={
            "n_predictions":
                "n_predictions_climate",
            "mae":
                "mae_climate",
            "rmse":
                "rmse_climate",
            "mase":
                "mase_climate",
        }
    )


    epidemiology = epidemiology[
        [
            "setting",
            "test_year",
            "horizon_weeks",
            "n_predictions_epidemiology",
            "mae_epidemiology",
            "rmse_epidemiology",
            "mase_epidemiology",
        ]
    ]


    climate = climate[
        [
            "setting",
            "test_year",
            "horizon_weeks",
            "n_predictions_climate",
            "mae_climate",
            "rmse_climate",
            "mase_climate",
        ]
    ]


    paired = epidemiology.merge(
        climate,
        on=[
            "setting",
            "test_year",
            "horizon_weeks",
        ],
        how="inner",
        validate="one_to_one",
    )


    if paired.empty:

        raise RuntimeError(
            f"No paired rows found for {family}."
        )


    if not (
        paired[
            "n_predictions_epidemiology"
        ]
        ==
        paired[
            "n_predictions_climate"
        ]
    ).all():

        raise RuntimeError(
            f"Prediction counts differ within {family} pairs."
        )


    paired[
        "family"
    ] = family


    paired[
        "epidemiology_model"
    ] = epidemiology_model


    paired[
        "climate_model"
    ] = climate_model


    # --------------------------------------------------------
    # DELTAS
    #
    # Negative = climate model has lower error
    # --------------------------------------------------------

    paired[
        "delta_mae"
    ] = (
        paired[
            "mae_climate"
        ]
        -
        paired[
            "mae_epidemiology"
        ]
    )


    paired[
        "delta_rmse"
    ] = (
        paired[
            "rmse_climate"
        ]
        -
        paired[
            "rmse_epidemiology"
        ]
    )


    paired[
        "delta_mase"
    ] = (
        paired[
            "mase_climate"
        ]
        -
        paired[
            "mase_epidemiology"
        ]
    )


    paired[
        "climate_improves_mae"
    ] = (
        paired[
            "delta_mae"
        ]
        <
        0
    )


    paired[
        "climate_improves_rmse"
    ] = (
        paired[
            "delta_rmse"
        ]
        <
        0
    )


    paired[
        "climate_improves_mase"
    ] = (
        paired[
            "delta_mase"
        ]
        <
        0
    )


    comparison_frames.append(
        paired
    )


by_setting = pd.concat(
    comparison_frames,
    ignore_index=True,
)


by_setting = by_setting.sort_values(
    [
        "family",
        "horizon_weeks",
        "setting",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE SETTING-LEVEL RESULTS
# ============================================================

by_setting_file = (
    TABLE_DIR
    / "climate_contribution_by_setting.csv"
)


by_setting.to_csv(
    by_setting_file,
    index=False,
)


# ============================================================
# SUMMARY ACROSS SETTINGS
# ============================================================

summary = (
    by_setting
    .groupby(
        [
            "family",
            "horizon_weeks",
        ],
        as_index=False,
    )
    .agg(
        settings_evaluated=(
            "setting",
            "nunique",
        ),
        median_delta_mae=(
            "delta_mae",
            "median",
        ),
        median_delta_rmse=(
            "delta_rmse",
            "median",
        ),
        median_delta_mase=(
            "delta_mase",
            "median",
        ),
        settings_climate_improves_mae=(
            "climate_improves_mae",
            "sum",
        ),
        settings_climate_improves_rmse=(
            "climate_improves_rmse",
            "sum",
        ),
        settings_climate_improves_mase=(
            "climate_improves_mase",
            "sum",
        ),
    )
)


summary[
    "fraction_climate_improves_mae"
] = (
    summary[
        "settings_climate_improves_mae"
    ]
    /
    summary[
        "settings_evaluated"
    ]
)


summary[
    "fraction_climate_improves_rmse"
] = (
    summary[
        "settings_climate_improves_rmse"
    ]
    /
    summary[
        "settings_evaluated"
    ]
)


summary[
    "fraction_climate_improves_mase"
] = (
    summary[
        "settings_climate_improves_mase"
    ]
    /
    summary[
        "settings_evaluated"
    ]
)


summary = summary.sort_values(
    [
        "horizon_weeks",
        "family",
    ]
).reset_index(
    drop=True
)


summary_file = (
    TABLE_DIR
    / "climate_contribution_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# TARGET-LEVEL PAIRED COMPARISON
# ============================================================

predictions = pd.read_csv(
    COMMON_PREDICTION_FILE
)


required_prediction_columns = {
    "setting",
    "test_year",
    "target_analysis_week",
    "horizon_weeks",
    "model",
    "actual",
    "predicted",
}


missing_prediction_columns = (
    required_prediction_columns
    - set(predictions.columns)
)


if missing_prediction_columns:

    raise ValueError(
        "Missing prediction columns: "
        + ", ".join(
            sorted(
                missing_prediction_columns
            )
        )
    )


for column in [
    "test_year",
    "target_analysis_week",
    "horizon_weeks",
    "actual",
    "predicted",
]:

    predictions[column] = pd.to_numeric(
        predictions[column],
        errors="coerce",
    )


target_frames = []


TARGET_KEYS = [
    "setting",
    "test_year",
    "target_analysis_week",
    "horizon_weeks",
]


for family, pair in MODEL_PAIRS.items():

    epidemiology_model = pair[
        "epidemiology"
    ]

    climate_model = pair[
        "climate"
    ]


    epidemiology = predictions[
        predictions[
            "model"
        ]
        ==
        epidemiology_model
    ][
        TARGET_KEYS
        +
        [
            "actual",
            "predicted",
        ]
    ].copy()


    climate = predictions[
        predictions[
            "model"
        ]
        ==
        climate_model
    ][
        TARGET_KEYS
        +
        [
            "actual",
            "predicted",
        ]
    ].copy()


    epidemiology = epidemiology.rename(
        columns={
            "actual":
                "actual_epidemiology",
            "predicted":
                "predicted_epidemiology",
        }
    )


    climate = climate.rename(
        columns={
            "actual":
                "actual_climate",
            "predicted":
                "predicted_climate",
        }
    )


    target_pair = epidemiology.merge(
        climate,
        on=TARGET_KEYS,
        how="inner",
        validate="one_to_one",
    )


    if target_pair.empty:

        raise RuntimeError(
            f"No target-level pairs found for {family}."
        )


    actual_difference = np.abs(
        target_pair[
            "actual_epidemiology"
        ]
        -
        target_pair[
            "actual_climate"
        ]
    )


    if (
        actual_difference
        >
        1e-9
    ).any():

        raise RuntimeError(
            f"Actual values disagree within {family}."
        )


    target_pair[
        "actual"
    ] = target_pair[
        "actual_epidemiology"
    ]


    target_pair[
        "absolute_error_epidemiology"
    ] = np.abs(
        target_pair[
            "predicted_epidemiology"
        ]
        -
        target_pair[
            "actual"
        ]
    )


    target_pair[
        "absolute_error_climate"
    ] = np.abs(
        target_pair[
            "predicted_climate"
        ]
        -
        target_pair[
            "actual"
        ]
    )


    target_pair[
        "delta_absolute_error"
    ] = (
        target_pair[
            "absolute_error_climate"
        ]
        -
        target_pair[
            "absolute_error_epidemiology"
        ]
    )


    target_pair[
        "squared_error_epidemiology"
    ] = (
        target_pair[
            "predicted_epidemiology"
        ]
        -
        target_pair[
            "actual"
        ]
    ) ** 2


    target_pair[
        "squared_error_climate"
    ] = (
        target_pair[
            "predicted_climate"
        ]
        -
        target_pair[
            "actual"
        ]
    ) ** 2


    target_pair[
        "delta_squared_error"
    ] = (
        target_pair[
            "squared_error_climate"
        ]
        -
        target_pair[
            "squared_error_epidemiology"
        ]
    )


    target_pair[
        "climate_lower_absolute_error"
    ] = (
        target_pair[
            "delta_absolute_error"
        ]
        <
        0
    )


    target_pair[
        "family"
    ] = family


    target_pair[
        "epidemiology_model"
    ] = epidemiology_model


    target_pair[
        "climate_model"
    ] = climate_model


    target_frames.append(
        target_pair
    )


paired_targets = pd.concat(
    target_frames,
    ignore_index=True,
)


paired_targets = paired_targets.sort_values(
    [
        "family",
        "horizon_weeks",
        "setting",
        "target_analysis_week",
    ]
).reset_index(
    drop=True
)


paired_target_file = (
    PREDICTION_DIR
    / "climate_contribution_paired_targets.csv"
)


paired_targets.to_csv(
    paired_target_file,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if by_setting.empty:

    raise RuntimeError(
        "Climate contribution setting-level output is empty."
    )


if summary.empty:

    raise RuntimeError(
        "Climate contribution summary is empty."
    )


if paired_targets.empty:

    raise RuntimeError(
        "Climate contribution target-level output is empty."
    )


if (
    by_setting[
        [
            "delta_mae",
            "delta_rmse",
            "delta_mase",
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "Missing delta metrics found."
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "CLIMATE CONTRIBUTION BY SETTING"
)

print(
    "=" * 80
)


print(
    by_setting[
        [
            "family",
            "setting",
            "horizon_weeks",
            "delta_mae",
            "delta_rmse",
            "delta_mase",
            "climate_improves_mae",
            "climate_improves_rmse",
            "climate_improves_mase",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "CLIMATE CONTRIBUTION SUMMARY"
)

print(
    "=" * 80
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\nInterpretation:"
)

print(
    "  Negative delta = climate-informed model has lower error."
)

print(
    "  Positive delta = climate-informed model has higher error."
)


print(
    "\nCLIMATE CONTRIBUTION ANALYSIS COMPLETE"
)


print(
    f"Saved: {by_setting_file}"
)

print(
    f"Saved: {summary_file}"
)

print(
    f"Saved: {paired_target_file}"
)