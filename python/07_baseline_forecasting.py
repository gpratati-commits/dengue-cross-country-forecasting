"""
Baseline dengue forecasting.

Forecast horizons:
- 1 week
- 2 weeks
- 4 weeks

Baselines:
1. Persistence
2. Seasonal naive
3. Previous-4-week mean

Evaluation:
The final source year of each setting is held out as an
out-of-sample test period.

Missing surveillance observations remain missing and are
never converted to zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/dengue_model_features.csv"
)

OUTPUT_DIR = Path(
    "outputs/tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found: {DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE
)

df = (
    df
    .sort_values(
        [
            "setting",
            "analysis_week"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# METRICS
# ============================================================

def mae(actual, predicted):

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


# ============================================================
# CREATE BASELINE PREDICTIONS
# ============================================================

HORIZONS = [
    1,
    2,
    4
]

parts = []


for setting, group in df.groupby(
    "setting",
    sort=False
):

    group = (
        group
        .sort_values(
            "analysis_week"
        )
        .copy()
    )

    # Forecast origin = current row t.
    # Persistence predicts future cases using cases at t.

    for horizon in HORIZONS:

        group[
            f"pred_persistence_h{horizon}"
        ] = group["cases"]

        # Seasonal naive forecast for t+h:
        # use cases observed 52 weeks before the target,
        # i.e. at t + h - 52.

        group[
            f"pred_seasonal_h{horizon}"
        ] = (
            group["cases"]
            .shift(
                52 - horizon
            )
        )

        # Use only information available at forecast origin.
        group[
            f"pred_mean4_h{horizon}"
        ] = (
            group["cases"]
            .rolling(
                window=4,
                min_periods=4
            )
            .mean()
        )

    parts.append(
        group
    )


df = pd.concat(
    parts,
    ignore_index=True
)


# ============================================================
# MASE SCALE
# ============================================================

def mase_scale(
    training
):

    training = (
        training
        .sort_values(
            "analysis_week"
        )
        .copy()
    )

    previous_cases = (
        training["cases"]
        .shift(1)
    )

    previous_week = (
        training["analysis_week"]
        .shift(1)
    )

    consecutive = (
        (
            training["analysis_week"]
            - previous_week
        )
        == 1
    )

    valid = (
        training["cases"].notna()
        & previous_cases.notna()
        & consecutive
    )

    differences = (
        training.loc[
            valid,
            "cases"
        ]
        .to_numpy()
        -
        previous_cases.loc[
            valid
        ]
        .to_numpy()
    )

    if len(differences) == 0:
        return np.nan

    return np.mean(
        np.abs(
            differences
        )
    )


# ============================================================
# EVALUATE BASELINES
# ============================================================

results = []


for setting, group in df.groupby(
    "setting"
):

    group = (
        group
        .sort_values(
            "analysis_week"
        )
        .copy()
    )

    test_year = int(
        group[
            "source_year"
        ].max()
    )

    training = group[
        group[
            "source_year"
        ]
        < test_year
    ].copy()

    test = group[
        group[
            "source_year"
        ]
        == test_year
    ].copy()

    scale = mase_scale(
        training
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


        for model_name, prediction in (
            models.items()
        ):

            evaluation = (
                test[
                    [
                        target,
                        prediction
                    ]
                ]
                .dropna()
            )

            if evaluation.empty:
                continue

            actual = (
                evaluation[
                    target
                ]
                .to_numpy()
            )

            predicted = (
                evaluation[
                    prediction
                ]
                .to_numpy()
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


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_file = (
    OUTPUT_DIR
    / "baseline_forecasting_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


print(
    "\nBASELINE FORECASTING RESULTS"
)

print(
    results_df.to_string(
        index=False
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
            "model"
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
        )
    )
)

summary_file = (
    OUTPUT_DIR
    / "baseline_forecasting_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


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