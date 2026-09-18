"""
Multi-horizon dengue baseline forecasting.

Baselines:
1. Persistence
2. Seasonal naive
3. Four-week moving average

Forecast horizons:
1, 2 and 4 weeks ahead.

Outputs:
- baseline_forecasting_results.csv
- baseline_forecasting_summary.csv
- baseline_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "data/processed/dengue_model_features.csv"
)

TABLE_DIR = Path(
    "outputs/tables"
)

PREDICTION_DIR = Path(
    "outputs/predictions"
)

HORIZONS = [
    1,
    2,
    4,
]

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================

def mae(
    actual,
    predicted
):
    """Mean absolute error."""

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


def rmse(
    actual,
    predicted
):
    """Root mean squared error."""

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    actual
                    - predicted
                )
                ** 2
            )
        )
    )


def mase_scale(
    training
):
    """
    Calculate one-step naive scale for MASE.
    """

    training = (
        training
        .sort_values(
            "analysis_week"
        )
        .copy()
    )

    previous_cases = (
        training[
            "cases"
        ]
        .shift(1)
    )

    previous_week = (
        training[
            "analysis_week"
        ]
        .shift(1)
    )

    valid = (
        training[
            "cases"
        ].notna()
        & previous_cases.notna()
        & (
            training[
                "analysis_week"
            ]
            - previous_week
        ).eq(1)
    )

    if not valid.any():
        return np.nan

    differences = (
        training.loc[
            valid,
            "cases"
        ].to_numpy(
            dtype=float
        )
        - previous_cases.loc[
            valid
        ].to_numpy(
            dtype=float
        )
    )

    scale = np.mean(
        np.abs(
            differences
        )
    )

    if (
        not np.isfinite(
            scale
        )
        or scale <= 0
    ):
        return np.nan

    return float(
        scale
    )


# ============================================================
# READ DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {DATA_FILE}"
    )

data = pd.read_csv(
    DATA_FILE
)

print(
    f"Rows loaded: {len(data):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = {
    "setting",
    "source_year",
    "source_week",
    "cases",
}

missing = sorted(
    required
    - set(
        data.columns
    )
)

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            missing
        )
    )


# ============================================================
# STANDARDISE CORE VARIABLES
# ============================================================

for column in [
    "source_year",
    "source_week",
    "cases",
]:
    data[
        column
    ] = pd.to_numeric(
        data[
            column
        ],
        errors="coerce"
    )


data = data.dropna(
    subset=[
        "setting",
        "source_year",
        "source_week",
    ]
).copy()


data[
    "source_year"
] = (
    data[
        "source_year"
    ]
    .astype(int)
)


data[
    "source_week"
] = (
    data[
        "source_week"
    ]
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
    .reset_index(
        drop=True
    )
)


# ============================================================
# WEEK LABEL
# ============================================================

if (
    "week_label"
    not in data.columns
):
    data[
        "week_label"
    ] = (
        data[
            "source_year"
        ]
        .astype(str)
        + "-W"
        + data[
            "source_week"
        ]
        .astype(str)
        .str.zfill(2)
    )


# ============================================================
# ANALYSIS WEEK
# ============================================================

if (
    "analysis_week"
    not in data.columns
):
    data[
        "analysis_week"
    ] = (
        data
        .groupby(
            "setting"
        )
        .cumcount()
        + 1
    )


data[
    "analysis_week"
] = pd.to_numeric(
    data[
        "analysis_week"
    ],
    errors="coerce"
)


data = data.dropna(
    subset=[
        "analysis_week"
    ]
).copy()


data[
    "analysis_week"
] = (
    data[
        "analysis_week"
    ]
    .astype(int)
)


# ============================================================
# CREATE TARGETS AND BASELINE FORECASTS
# ============================================================

for horizon in HORIZONS:

    target = (
        f"target_cases_h{horizon}"
    )

    if (
        target
        not in data.columns
    ):
        data[
            target
        ] = (
            data
            .groupby(
                "setting"
            )[
                "cases"
            ]
            .shift(
                -horizon
            )
        )


    # --------------------------------------------------------
    # PERSISTENCE
    # Forecast future cases using cases observed now.
    # --------------------------------------------------------

    data[
        f"pred_persistence_h{horizon}"
    ] = data[
        "cases"
    ]


    # --------------------------------------------------------
    # SEASONAL NAIVE
    # Forecast target t+h using the corresponding value
    # approximately 52 weeks before the target.
    # --------------------------------------------------------

    seasonal_lag = (
        52
        - horizon
    )

    data[
        f"pred_seasonal_h{horizon}"
    ] = (
        data
        .groupby(
            "setting"
        )[
            "cases"
        ]
        .shift(
            seasonal_lag
        )
    )


    # --------------------------------------------------------
    # FOUR-WEEK MOVING AVERAGE
    # Uses information available at forecast origin.
    # --------------------------------------------------------

    data[
        f"pred_mean4_h{horizon}"
    ] = (
        data
        .groupby(
            "setting"
        )[
            "cases"
        ]
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
    data[
        "setting"
    ]
    .dropna()
    .unique()
)


for setting in settings:

    group = (
        data.loc[
            data[
                "setting"
            ]
            == setting
        ]
        .sort_values(
            "analysis_week"
        )
        .copy()
    )


    observed_years = sorted(
        group.loc[
            group[
                "cases"
            ].notna(),
            "source_year"
        ]
        .dropna()
        .astype(int)
        .unique()
    )


    if (
        len(
            observed_years
        )
        < 2
    ):
        print(
            f"Skipping {setting}: "
            "fewer than two observed years."
        )

        continue


    test_year = (
        observed_years[
            -1
        ]
    )


    training = (
        group.loc[
            group[
                "source_year"
            ]
            < test_year
        ]
        .copy()
    )


    scale = mase_scale(
        training
    )


    print(
        f"\n{setting}: "
        f"test={test_year}"
    )


    for horizon in HORIZONS:

        target = (
            f"target_cases_h{horizon}"
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

            evaluation = (
                group.loc[
                    group[
                        "source_year"
                    ]
                    == test_year,
                    [
                        "setting",
                        "source_year",
                        "source_week",
                        "week_label",
                        "analysis_week",
                        target,
                        prediction_column,
                    ],
                ]
                .rename(
                    columns={
                        target:
                            "actual",

                        prediction_column:
                            "predicted",
                    }
                )
                .dropna(
                    subset=[
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


            actual = (
                evaluation[
                    "actual"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            predicted = (
                evaluation[
                    "predicted"
                ]
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
                pd.notna(
                    scale
                )
                and scale > 0
            ):
                model_mase = (
                    model_mae
                    / scale
                )

            else:
                model_mase = np.nan


            # ------------------------------------------------
            # SUMMARY RESULT
            # ------------------------------------------------

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
                        len(
                            evaluation
                        ),

                    "mae":
                        model_mae,

                    "rmse":
                        model_rmse,

                    "mase":
                        model_mase,
                }
            )


            # ------------------------------------------------
            # PREDICTION-LEVEL RESULT
            # ------------------------------------------------

            detail = (
                evaluation
                .copy()
            )


            detail[
                "test_year"
            ] = test_year


            detail[
                "target_analysis_week"
            ] = (
                detail[
                    "analysis_week"
                ]
                + horizon
            )


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
# CREATE RESULTS DATAFRAME
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
# CREATE PREDICTION DATAFRAME
# ============================================================

if (
    len(
        prediction_frames
    )
    == 0
):

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
            "analysis_week",
        ]
    )
    .reset_index(
        drop=True
    )
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