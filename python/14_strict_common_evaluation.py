"""
Strict common-cohort evaluation of dengue forecasting models.

The purpose of this script is to compare all forecast models only on
observations for which every model has a prediction.

Models:
    Baselines
        - persistence
        - seasonal_naive
        - moving_average_4

    Negative binomial
        - negative_binomial_epidemiology
        - negative_binomial_climate

    XGBoost
        - xgboost_epidemiology
        - xgboost_climate

Metrics:
    - MAE
    - RMSE
    - MASE

Outputs:
    outputs/tables/strict_common_cohort_coverage.csv
    outputs/tables/strict_common_evaluation_by_setting.csv
    outputs/tables/strict_common_evaluation_summary.csv
    outputs/predictions/strict_common_cohort_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PREDICTION_DIR = Path("outputs/predictions")
TABLE_DIR = Path("outputs/tables")

FEATURE_FILE = Path(
    "data/processed/dengue_model_features.csv"
)

BASELINE_FILE = (
    PREDICTION_DIR
    / "baseline_predictions.csv"
)

NB_FILE = (
    PREDICTION_DIR
    / "negative_binomial_predictions.csv"
)

XGB_FILE = (
    PREDICTION_DIR
    / "xgboost_predictions.csv"
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
# EXPECTED MODELS
# ============================================================

EXPECTED_MODELS = [
    "persistence",
    "seasonal_naive",
    "moving_average_4",
    "negative_binomial_epidemiology",
    "negative_binomial_climate",
    "xgboost_epidemiology",
    "xgboost_climate",
]


EXPECTED_HORIZONS = {
    1,
    2,
    4,
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_PREDICTION_COLUMNS = [
    "setting",
    "test_year",
    "source_year",
    "source_week",
    "analysis_week",
    "target_year",
    "target_analysis_week",
    "horizon_weeks",
    "model",
    "actual",
    "predicted",
]


# ============================================================
# READ AND VALIDATE A PREDICTION FILE
# ============================================================

def read_prediction_file(
    path,
    family,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )


    df = pd.read_csv(
        path
    )


    missing_columns = [
        column
        for column in REQUIRED_PREDICTION_COLUMNS
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"{family}: missing columns: "
            + ", ".join(missing_columns)
        )


    numeric_columns = [
        "test_year",
        "source_year",
        "source_week",
        "analysis_week",
        "target_year",
        "target_analysis_week",
        "horizon_weeks",
        "actual",
        "predicted",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    if df["actual"].isna().any():

        raise ValueError(
            f"{family}: missing actual values."
        )


    if df["predicted"].isna().any():

        raise ValueError(
            f"{family}: missing predicted values."
        )


    if not np.isfinite(
        df["actual"].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            f"{family}: non-finite actual values."
        )


    if not np.isfinite(
        df["predicted"].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            f"{family}: non-finite predictions."
        )


    bad_target_year = (
        df["target_year"]
        !=
        df["test_year"]
    ).sum()


    if bad_target_year != 0:

        raise ValueError(
            f"{family}: "
            f"{bad_target_year} rows have "
            "target_year != test_year."
        )


    bad_target_week = (
        df["target_analysis_week"]
        !=
        (
            df["analysis_week"]
            +
            df["horizon_weeks"]
        )
    ).sum()


    if bad_target_week != 0:

        raise ValueError(
            f"{family}: "
            f"{bad_target_week} rows have "
            "incorrect target_analysis_week."
        )


    return df.copy()


# ============================================================
# READ FORECASTS
# ============================================================

baseline = read_prediction_file(
    BASELINE_FILE,
    "baseline",
)

negative_binomial = read_prediction_file(
    NB_FILE,
    "negative_binomial",
)

xgboost = read_prediction_file(
    XGB_FILE,
    "xgboost",
)


# ============================================================
# STANDARDISE MODEL NAMES
# ============================================================

nb_name_map = {
    "epidemiology_only":
        "negative_binomial_epidemiology",

    "climate_informed":
        "negative_binomial_climate",
}


negative_binomial["model"] = (
    negative_binomial["model"]
    .replace(
        nb_name_map
    )
)


# ============================================================
# COMBINE ALL PREDICTIONS
# ============================================================

all_predictions = pd.concat(
    [
        baseline,
        negative_binomial,
        xgboost,
    ],
    ignore_index=True,
)


observed_models = sorted(
    all_predictions[
        "model"
    ]
    .dropna()
    .unique()
)


missing_models = sorted(
    set(EXPECTED_MODELS)
    -
    set(observed_models)
)


unexpected_models = sorted(
    set(observed_models)
    -
    set(EXPECTED_MODELS)
)


if missing_models:

    raise RuntimeError(
        "Expected models missing: "
        + ", ".join(missing_models)
    )


if unexpected_models:

    raise RuntimeError(
        "Unexpected models found: "
        + ", ".join(unexpected_models)
    )


observed_horizons = set(
    all_predictions[
        "horizon_weeks"
    ]
    .dropna()
    .astype(int)
    .unique()
)


if observed_horizons != EXPECTED_HORIZONS:

    raise RuntimeError(
        "Observed horizons do not match "
        f"{sorted(EXPECTED_HORIZONS)}. "
        f"Observed: {sorted(observed_horizons)}"
    )


# ============================================================
# FORECAST KEY
#
# Every model must forecast the same target observation.
# ============================================================

KEY_COLUMNS = [
    "setting",
    "test_year",
    "target_year",
    "target_analysis_week",
    "horizon_weeks",
]


# ============================================================
# CHECK DUPLICATE MODEL FORECASTS
# ============================================================

duplicate_model_rows = (
    all_predictions
    .duplicated(
        subset=(
            KEY_COLUMNS
            +
            ["model"]
        )
    )
    .sum()
)


if duplicate_model_rows != 0:

    raise RuntimeError(
        "Duplicate model forecasts found: "
        f"{duplicate_model_rows}"
    )


# ============================================================
# CHECK THAT ACTUAL VALUES AGREE
# ============================================================

actual_consistency = (
    all_predictions
    .groupby(
        KEY_COLUMNS
    )["actual"]
    .agg(
        [
            "min",
            "max",
        ]
    )
)


actual_consistency[
    "difference"
] = (
    actual_consistency["max"]
    -
    actual_consistency["min"]
)


inconsistent_actual = (
    actual_consistency[
        "difference"
    ]
    >
    1e-9
).sum()


if inconsistent_actual != 0:

    raise RuntimeError(
        "Actual values disagree across models "
        f"for {inconsistent_actual} forecast targets."
    )


# ============================================================
# IDENTIFY STRICT COMMON COHORT
#
# A forecast target is retained only if all seven models
# generated a prediction for that exact target.
# ============================================================

model_counts = (
    all_predictions
    .groupby(
        KEY_COLUMNS
    )["model"]
    .nunique()
    .rename(
        "models_available"
    )
    .reset_index()
)


common_keys = model_counts[
    model_counts[
        "models_available"
    ]
    ==
    len(EXPECTED_MODELS)
].copy()


if common_keys.empty:

    raise RuntimeError(
        "No forecast targets contain predictions "
        "from all seven models."
    )


# ============================================================
# CREATE STRICT COMMON PREDICTION TABLE
# ============================================================

common_predictions = (
    all_predictions
    .merge(
        common_keys[
            KEY_COLUMNS
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
)


# ============================================================
# CHECK EVERY COMMON TARGET HAS ALL SEVEN MODELS
# ============================================================

common_model_counts = (
    common_predictions
    .groupby(
        KEY_COLUMNS
    )["model"]
    .nunique()
)


if not (
    common_model_counts
    ==
    len(EXPECTED_MODELS)
).all():

    raise RuntimeError(
        "Strict common cohort construction failed."
    )


# ============================================================
# COMMON-COHORT COVERAGE
# ============================================================

coverage = (
    common_keys
    .groupby(
        [
            "setting",
            "horizon_weeks",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size":
                "common_forecast_targets"
        }
    )
)


coverage_file = (
    TABLE_DIR
    /
    "strict_common_cohort_coverage.csv"
)


coverage.to_csv(
    coverage_file,
    index=False,
)


# ============================================================
# MASE SCALE FROM HISTORICAL DATA ONLY
# ============================================================

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Missing feature file: {FEATURE_FILE}"
    )


history = pd.read_csv(
    FEATURE_FILE
)


history_required = [
    "setting",
    "source_year",
    "analysis_week",
    "cases",
]


history_missing = [
    column
    for column in history_required
    if column not in history.columns
]


if history_missing:

    raise ValueError(
        "Feature file is missing columns: "
        + ", ".join(history_missing)
    )


for column in [
    "source_year",
    "analysis_week",
    "cases",
]:

    history[column] = pd.to_numeric(
        history[column],
        errors="coerce",
    )


# ============================================================
# BUILD SCALE FOR EACH SETTING / TEST YEAR
# ============================================================

scale_rows = []


setting_test_years = (
    common_predictions[
        [
            "setting",
            "test_year",
        ]
    ]
    .drop_duplicates()
)


for row in setting_test_years.itertuples(
    index=False
):

    setting_name = row.setting

    test_year = int(
        row.test_year
    )


    training_history = history[
        (
            history["setting"]
            ==
            setting_name
        )
        &
        (
            history["source_year"]
            <
            test_year
        )
    ].copy()


    training_history = (
        training_history
        .dropna(
            subset=[
                "analysis_week",
                "cases",
            ]
        )
        .sort_values(
            "analysis_week"
        )
    )


    previous_week = (
        training_history[
            "analysis_week"
        ]
        .shift(1)
    )


    previous_cases = (
        training_history[
            "cases"
        ]
        .shift(1)
    )


    consecutive = (
        training_history[
            "analysis_week"
        ]
        -
        previous_week
        ==
        1
    )


    differences = (
        training_history.loc[
            consecutive,
            "cases",
        ]
        .to_numpy(
            dtype=float
        )
        -
        previous_cases.loc[
            consecutive
        ]
        .to_numpy(
            dtype=float
        )
    )


    if len(differences) == 0:

        scale = np.nan

    else:

        scale = float(
            np.mean(
                np.abs(
                    differences
                )
            )
        )


    if (
        not np.isfinite(scale)
        or scale <= 0
    ):

        scale = np.nan


    scale_rows.append(
        {
            "setting":
                setting_name,

            "test_year":
                test_year,

            "mase_scale":
                scale,
        }
    )


scale_df = pd.DataFrame(
    scale_rows
)


common_predictions = (
    common_predictions
    .merge(
        scale_df,
        on=[
            "setting",
            "test_year",
        ],
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# ROW-LEVEL ERRORS
# ============================================================

common_predictions[
    "error"
] = (
    common_predictions[
        "predicted"
    ]
    -
    common_predictions[
        "actual"
    ]
)


common_predictions[
    "absolute_error"
] = np.abs(
    common_predictions[
        "error"
    ]
)


common_predictions[
    "squared_error"
] = (
    common_predictions[
        "error"
    ]
    ** 2
)


common_predictions[
    "absolute_scaled_error"
] = (
    common_predictions[
        "absolute_error"
    ]
    /
    common_predictions[
        "mase_scale"
    ]
)


# ============================================================
# SAVE COMMON PREDICTIONS
# ============================================================

common_prediction_file = (
    PREDICTION_DIR
    /
    "strict_common_cohort_predictions.csv"
)


common_predictions.to_csv(
    common_prediction_file,
    index=False,
)


# ============================================================
# SETTING-LEVEL EVALUATION
# ============================================================

evaluation_rows = []


group_columns = [
    "setting",
    "test_year",
    "horizon_weeks",
    "model",
]


for key, group in common_predictions.groupby(
    group_columns,
    sort=True,
):

    (
        setting_name,
        test_year,
        horizon,
        model_name,
    ) = key


    actual = group[
        "actual"
    ].to_numpy(
        dtype=float
    )


    predicted = group[
        "predicted"
    ].to_numpy(
        dtype=float
    )


    error = (
        predicted
        -
        actual
    )


    model_mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )


    model_rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )


    scale_values = (
        group[
            "mase_scale"
        ]
        .dropna()
        .unique()
    )


    if len(scale_values) == 1:

        scale = float(
            scale_values[0]
        )

    else:

        scale = np.nan


    if (
        np.isfinite(scale)
        and scale > 0
    ):

        model_mase = (
            model_mae
            /
            scale
        )

    else:

        model_mase = np.nan


    evaluation_rows.append(
        {
            "setting":
                setting_name,

            "test_year":
                int(test_year),

            "horizon_weeks":
                int(horizon),

            "model":
                model_name,

            "n_predictions":
                len(group),

            "mae":
                model_mae,

            "rmse":
                model_rmse,

            "mase":
                model_mase,
        }
    )


evaluation_df = pd.DataFrame(
    evaluation_rows
)


evaluation_df = (
    evaluation_df
    .sort_values(
        [
            "horizon_weeks",
            "setting",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


evaluation_file = (
    TABLE_DIR
    /
    "strict_common_evaluation_by_setting.csv"
)


evaluation_df.to_csv(
    evaluation_file,
    index=False,
)


# ============================================================
# SUMMARY ACROSS SETTINGS
# ============================================================

summary_metrics = (
    evaluation_df
    .groupby(
        [
            "horizon_weeks",
            "model",
        ],
        as_index=False,
    )
    .agg(
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
        settings_evaluated=(
            "setting",
            "nunique",
        ),
        total_predictions=(
            "n_predictions",
            "sum",
        ),
    )
)


summary_metrics = (
    summary_metrics
    .sort_values(
        [
            "horizon_weeks",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


summary_file = (
    TABLE_DIR
    /
    "strict_common_evaluation_summary.csv"
)


summary_metrics.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

models_per_horizon = (
    summary_metrics
    .groupby(
        "horizon_weeks"
    )["model"]
    .nunique()
)


if not (
    models_per_horizon
    ==
    len(EXPECTED_MODELS)
).all():

    raise RuntimeError(
        "Not all seven models were evaluated "
        "at every forecast horizon."
    )


horizons_in_summary = set(
    summary_metrics[
        "horizon_weeks"
    ]
    .astype(int)
    .unique()
)


if horizons_in_summary != EXPECTED_HORIZONS:

    raise RuntimeError(
        "Strict evaluation is missing "
        "one or more forecast horizons."
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "STRICT COMMON-COHORT COVERAGE"
)

print(
    "=" * 72
)


print(
    coverage.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 72
)

print(
    "STRICT COMMON-COHORT MODEL EVALUATION"
)

print(
    "=" * 72
)


print(
    summary_metrics.to_string(
        index=False
    )
)


print(
    "\nStrict common forecast targets:",
    len(common_keys),
)


print(
    "Strict common prediction rows:",
    len(common_predictions),
)


print(
    "Settings retained:",
    sorted(
        common_predictions[
            "setting"
        ]
        .unique()
    ),
)


print(
    "Horizons retained:",
    sorted(
        common_predictions[
            "horizon_weeks"
        ]
        .astype(int)
        .unique()
    ),
)


print(
    "Models retained:",
    sorted(
        common_predictions[
            "model"
        ]
        .unique()
    ),
)


print(
    "\nSTRICT COMMON-COHORT EVALUATION COMPLETE"
)


print(
    f"Saved: {coverage_file}"
)

print(
    f"Saved: {evaluation_file}"
)

print(
    f"Saved: {summary_file}"
)

print(
    f"Saved: {common_prediction_file}"
)