"""
Leakage-safe multi-horizon dengue baseline forecasting.

Models:
- persistence
- seasonal_naive
- moving_average_4

Horizons:
- 1, 2, and 4 weeks ahead

Important:
The held-out test cohort is defined using TARGET YEAR, not source year.
This prevents end-of-year forecasts from being assigned to the wrong
evaluation period.

Outputs:
- outputs/tables/baseline_forecasting_results.csv
- outputs/tables/baseline_forecasting_summary.csv
- outputs/predictions/baseline_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/dengue_model_features.csv")
TABLE_DIR = Path("outputs/tables")
PREDICTION_DIR = Path("outputs/predictions")

HORIZONS = [1, 2, 4]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.mean(
            np.abs(actual - predicted)
        )
    )


def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def mase_scale(training):
    """
    One-step naive MASE denominator using only consecutive
    observed weeks before the held-out test year.
    """

    training = (
        training
        .sort_values("analysis_week")
        .copy()
    )

    previous_cases = training["cases"].shift(1)
    previous_week = training["analysis_week"].shift(1)

    valid = (
        training["cases"].notna()
        & previous_cases.notna()
        & (
            training["analysis_week"]
            - previous_week
        ).eq(1)
    )

    if not valid.any():
        return np.nan

    differences = (
        training.loc[
            valid,
            "cases"
        ].to_numpy(dtype=float)
        - previous_cases.loc[
            valid
        ].to_numpy(dtype=float)
    )

    scale = np.mean(
        np.abs(differences)
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        return np.nan

    return float(scale)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Missing processed data file: {DATA_FILE}"
    )

data = pd.read_csv(
    DATA_FILE
)

print(
    f"Rows loaded: {len(data):,}"
)


required_columns = {
    "setting",
    "source_year",
    "source_week",
    "cases",
}

missing_columns = sorted(
    required_columns
    - set(data.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CLEAN CORE VARIABLES
# ============================================================

for column in [
    "source_year",
    "source_week",
    "cases",
]:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data = data.dropna(
    subset=[
        "setting",
        "source_year",
        "source_week",
    ]
).copy()


data["source_year"] = (
    data["source_year"]
    .astype(int)
)

data["source_week"] = (
    data["source_week"]
    .astype(int)
)


data = (
    data
    .sort_values(
        [
            "setting",
            "source_year",
            "source_week",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# WEEK LABEL AND ANALYSIS-WEEK INDEX
# ============================================================

if "week_label" not in data.columns:
    data["week_label"] = (
        data["source_year"].astype(str)
        + "-W"
        + data["source_week"]
        .astype(str)
        .str.zfill(2)
    )


if "analysis_week" not in data.columns:
    data["analysis_week"] = (
        data
        .groupby(
            "setting",
            sort=False
        )
        .cumcount()
        + 1
    )


data["analysis_week"] = pd.to_numeric(
    data["analysis_week"],
    errors="coerce"
)


data = data.dropna(
    subset=[
        "analysis_week"
    ]
).copy()


data["analysis_week"] = (
    data["analysis_week"]
    .astype(int)
)


data = (
    data
    .sort_values(
        [
            "setting",
            "analysis_week",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# CREATE TARGETS AND BASELINE FORECASTS
# ============================================================

for horizon in HORIZONS:

    target_cases = (
        f"target_cases_h{horizon}"
    )

    target_year = (
        f"target_year_h{horizon}"
    )

    target_week = (
        f"target_week_h{horizon}"
    )

    target_analysis_week = (
        f"target_analysis_week_h{horizon}"
    )


    grouped = data.groupby(
        "setting",
        sort=False
    )


    # --------------------------------------------------------
    # FUTURE OUTCOME AND TRUE TARGET-TIME METADATA
    # --------------------------------------------------------

    data[target_cases] = (
        grouped["cases"]
        .shift(-horizon)
    )


    data[target_year] = (
        grouped["source_year"]
        .shift(-horizon)
    )


    data[target_week] = (
        grouped["source_week"]
        .shift(-horizon)
    )


    data[target_analysis_week] = (
        grouped["analysis_week"]
        .shift(-horizon)
    )


    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    data[
        f"pred_persistence_h{horizon}"
    ] = data["cases"]


    # --------------------------------------------------------
    # SEASONAL NAIVE
    #
    # Forecast y_(t+h) using y_(t+h-52).
    # At forecast origin t this equals a lag of (52-h).
    # --------------------------------------------------------

    seasonal_lag = (
        52
        - horizon
    )


    data[
        f"pred_seasonal_h{horizon}"
    ] = (
        grouped["cases"]
        .shift(seasonal_lag)
    )


    # --------------------------------------------------------
    # FOUR-WEEK MOVING AVERAGE
    #
    # Uses information available at the forecast origin.
    # --------------------------------------------------------

    data[
        f"pred_mean4_h{horizon}"
    ] = (
        grouped["cases"]
        .transform(
            lambda series: (
                series
                .rolling(
                    window=4,
                    min_periods=4
                )
                .mean()
            )
        )
    )


# ============================================================
# STORAGE
# ============================================================

results = []

prediction_frames = []


# ============================================================
# FORECAST EVALUATION
# ============================================================

settings = sorted(
    data["setting"]
    .dropna()
    .unique()
)


for setting in settings:

    group = (
        data.loc[
            data["setting"]
            == setting
        ]
        .sort_values(
            "analysis_week"
        )
        .copy()
    )


    observed_years = sorted(
        group.loc[
            group["cases"].notna(),
            "source_year"
        ]
        .dropna()
        .astype(int)
        .unique()
    )


    if len(observed_years) < 2:

        print(
            f"Skipping {setting}: "
            "fewer than two observed years."
        )

        continue


    test_year = (
        observed_years[-1]
    )


    # --------------------------------------------------------
    # MASE DENOMINATOR
    #
    # Uses only observed outcomes before the held-out test year.
    # --------------------------------------------------------

    scale_training = (
        group.loc[
            group["source_year"]
            < test_year
        ]
        .copy()
    )


    scale = mase_scale(
        scale_training
    )


    print(
        f"\n{setting}: "
        f"test target year={test_year}"
    )


    for horizon in HORIZONS:

        target_cases = (
            f"target_cases_h{horizon}"
        )

        target_year = (
            f"target_year_h{horizon}"
        )

        target_week = (
            f"target_week_h{horizon}"
        )

        target_analysis_week = (
            f"target_analysis_week_h{horizon}"
        )


        models = {
            "persistence":
                f"pred_persistence_h{horizon}",

            "seasonal_naive":
                f"pred_seasonal_h{horizon}",

            "moving_average_4":
                f"pred_mean4_h{horizon}",
        }


        for (
            model_name,
            prediction_column
        ) in models.items():


            # =================================================
            # LEAKAGE-SAFE TEST SPLIT
            #
            # IMPORTANT:
            # Selection is based on TARGET YEAR,
            # not forecast-origin/source year.
            # =================================================

            evaluation = (
                group.loc[
                    group[target_year]
                    == test_year,
                    [
                        "setting",
                        "source_year",
                        "source_week",
                        "week_label",
                        "analysis_week",
                        target_year,
                        target_week,
                        target_analysis_week,
                        target_cases,
                        prediction_column,
                    ],
                ]
                .rename(
                    columns={
                        target_year:
                            "target_year",

                        target_week:
                            "target_week",

                        target_analysis_week:
                            "target_analysis_week",

                        target_cases:
                            "actual",

                        prediction_column:
                            "predicted",
                    }
                )
                .dropna(
                    subset=[
                        "target_year",
                        "target_week",
                        "target_analysis_week",
                        "actual",
                        "predicted",
                    ]
                )
                .copy()
            )


            if evaluation.empty:

                print(
                    f"Skipping {setting}, "
                    f"horizon={horizon}, "
                    f"model={model_name}: "
                    "no complete test rows."
                )

                continue


            evaluation[
                "target_year"
            ] = (
                evaluation[
                    "target_year"
                ]
                .astype(int)
            )


            evaluation[
                "target_week"
            ] = (
                evaluation[
                    "target_week"
                ]
                .astype(int)
            )


            evaluation[
                "target_analysis_week"
            ] = (
                evaluation[
                    "target_analysis_week"
                ]
                .astype(int)
            )


            actual = (
                evaluation["actual"]
                .to_numpy(
                    dtype=float
                )
            )


            predicted = (
                evaluation["predicted"]
                .to_numpy(
                    dtype=float
                )
            )


            model_mae = mae(
                actual,
                predicted
            )


            model_rmse = rmse(
                actual,
                predicted
            )


            if (
                pd.notna(scale)
                and scale > 0
            ):

                model_mase = (
                    model_mae
                    / scale
                )

            else:

                model_mase = (
                    np.nan
                )


            # =================================================
            # SUMMARY RESULTS
            # =================================================

            results.append(
                {
                    "setting":
                        setting,

                    "test_year":
                        test_year,

                    "horizon_weeks":
                        horizon,

                    "model":
                        model_name,

                    "n_predictions":
                        len(evaluation),

                    "mae":
                        model_mae,

                    "rmse":
                        model_rmse,

                    "mase":
                        model_mase,
                }
            )


            # =================================================
            # PREDICTION-LEVEL RESULTS
            # =================================================

            detail = (
                evaluation
                .copy()
            )


            detail[
                "test_year"
            ] = test_year


            detail[
                "horizon_weeks"
            ] = horizon


            detail[
                "model"
            ] = model_name


            detail = detail[
                [
                    "setting",
                    "test_year",
                    "source_year",
                    "source_week",
                    "week_label",
                    "analysis_week",
                    "target_year",
                    "target_week",
                    "target_analysis_week",
                    "horizon_weeks",
                    "model",
                    "actual",
                    "predicted",
                ]
            ]


            prediction_frames.append(
                detail
            )


# ============================================================
# BUILD RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No baseline forecasts were "
        "successfully evaluated."
    )


results_df = (
    results_df
    .sort_values(
        [
            "setting",
            "horizon_weeks",
            "model",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# BUILD PREDICTION TABLE
# ============================================================

if not prediction_frames:

    raise RuntimeError(
        "No baseline prediction-level "
        "rows were generated."
    )


predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)


predictions_df = (
    predictions_df
    .sort_values(
        [
            "setting",
            "horizon_weeks",
            "model",
            "target_analysis_week",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# DUPLICATE SAFETY CHECK
# ============================================================

duplicate_key = [
    "setting",
    "analysis_week",
    "target_analysis_week",
    "horizon_weeks",
    "model",
]


duplicate_count = int(
    predictions_df
    .duplicated(
        subset=duplicate_key
    )
    .sum()
)


if duplicate_count > 0:

    raise RuntimeError(
        "Duplicate baseline forecast rows detected: "
        f"{duplicate_count}"
    )


# ============================================================
# TARGET-YEAR SAFETY CHECK
# ============================================================

wrong_target_year = int(
    (
        predictions_df[
            "target_year"
        ]
        != predictions_df[
            "test_year"
        ]
    )
    .sum()
)


if wrong_target_year > 0:

    raise RuntimeError(
        "Leakage-safe test-cohort check failed: "
        f"{wrong_target_year} rows have "
        "target_year != test_year."
    )


# ============================================================
# SUMMARY ACROSS SETTINGS
# ============================================================

summary = (
    results_df
    .groupby(
        [
            "horizon_weeks",
            "model",
        ],
        as_index=False
    )
    .agg(
        median_mae=(
            "mae",
            "median"
        ),

        median_rmse=(
            "rmse",
            "median"
        ),

        median_mase=(
            "mase",
            "median"
        ),

        settings_evaluated=(
            "setting",
            "nunique"
        ),
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_file = (
    TABLE_DIR
    / "baseline_forecasting_results.csv"
)


summary_file = (
    TABLE_DIR
    / "baseline_forecasting_summary.csv"
)


prediction_file = (
    PREDICTION_DIR
    / "baseline_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


summary.to_csv(
    summary_file,
    index=False
)


predictions_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\nMEDIAN PERFORMANCE ACROSS SETTINGS"
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\nBASELINE PREDICTION DIAGNOSTICS"
)


print(
    "Rows:",
    len(predictions_df)
)


print(
    "Settings:",
    sorted(
        predictions_df[
            "setting"
        ]
        .unique()
    )
)


print(
    "Horizons:",
    sorted(
        predictions_df[
            "horizon_weeks"
        ]
        .unique()
    )
)


print(
    "Models:",
    sorted(
        predictions_df[
            "model"
        ]
        .unique()
    )
)


print(
    "Missing actual:",
    int(
        predictions_df[
            "actual"
        ]
        .isna()
        .sum()
    )
)


print(
    "Missing predicted:",
    int(
        predictions_df[
            "predicted"
        ]
        .isna()
        .sum()
    )
)


print(
    "Duplicate forecast rows:",
    duplicate_count
)


print(
    "Rows with target_year != test_year:",
    wrong_target_year
)


print(
    "\nBASELINE FORECASTING COMPLETE"
)


print(
    f"Saved: {results_file}"
)


print(
    f"Saved: {summary_file}"
)


print(
    f"Saved: {prediction_file}"
)