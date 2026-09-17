"""
Compare all dengue forecasting models on common
setting/horizon combinations.

Models compared:
- persistence
- seasonal naive
- 4-week moving average
- negative-binomial epidemiology-only
- negative-binomial climate-informed
- XGBoost epidemiology-only
- XGBoost climate-informed

The comparison is restricted to setting/horizon combinations
available across baseline, negative-binomial and XGBoost analyses.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

TABLE_DIR = Path("outputs/tables")

BASELINE_FILE = (
    TABLE_DIR
    / "baseline_forecasting_results.csv"
)

NB_FILE = (
    TABLE_DIR
    / "negative_binomial_results.csv"
)

XGB_FILE = (
    TABLE_DIR
    / "xgboost_forecasting_results.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    BASELINE_FILE,
    NB_FILE,
    XGB_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required results file not found: {file_path}"
        )


# ============================================================
# LOAD RESULTS
# ============================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

nb = pd.read_csv(
    NB_FILE
)

xgb = pd.read_csv(
    XGB_FILE
)


print(
    "\nRows loaded:"
)

print(
    f"Baseline: {len(baseline):,}"
)

print(
    f"Negative binomial: {len(nb):,}"
)

print(
    f"XGBoost: {len(xgb):,}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = {
    "setting",
    "horizon_weeks",
    "model",
    "mae",
    "rmse",
    "mase",
}


for name, data in {

    "baseline":
        baseline,

    "negative_binomial":
        nb,

    "xgboost":
        xgb,

}.items():

    missing = (
        required
        - set(data.columns)
    )

    if missing:

        raise KeyError(
            f"{name} results are missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# STANDARDIZE MODEL NAMES
# ============================================================

baseline = baseline.copy()

baseline[
    "model_family"
] = "baseline"

baseline[
    "model_standard"
] = baseline[
    "model"
].astype(str)


nb = nb.copy()

nb[
    "model_family"
] = "negative_binomial"

nb[
    "model_standard"
] = (
    "negative_binomial_"
    + nb[
        "model"
    ].astype(str)
)


xgb = xgb.copy()

xgb[
    "model_family"
] = "xgboost"

xgb[
    "model_standard"
] = xgb[
    "model"
].astype(str)


# ============================================================
# COMMON SETTING / HORIZON COMBINATIONS
# ============================================================

baseline_keys = (

    baseline[
        [
            "setting",
            "horizon_weeks",
        ]
    ]

    .drop_duplicates()
)


nb_keys = (

    nb[
        [
            "setting",
            "horizon_weeks",
        ]
    ]

    .drop_duplicates()
)


xgb_keys = (

    xgb[
        [
            "setting",
            "horizon_weeks",
        ]
    ]

    .drop_duplicates()
)


common_keys = (

    baseline_keys

    .merge(
        nb_keys,
        on=[
            "setting",
            "horizon_weeks",
        ],
        how="inner",
    )

    .merge(
        xgb_keys,
        on=[
            "setting",
            "horizon_weeks",
        ],
        how="inner",
    )
)


if common_keys.empty:

    raise RuntimeError(
        "No common setting/horizon combinations "
        "were found across the model families."
    )


print(
    "\nCOMMON SETTING/HORIZON COMBINATIONS"
)

print(
    common_keys
    .sort_values(
        [
            "setting",
            "horizon_weeks",
        ]
    )
    .to_string(
        index=False
    )
)


print(
    "\nSettings represented:",
    common_keys[
        "setting"
    ].nunique()
)


print(
    "Setting/horizon combinations:",
    len(common_keys)
)


# ============================================================
# RESTRICT RESULTS TO COMMON COHORT
# ============================================================

baseline_common = baseline.merge(
    common_keys,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="inner",
)


nb_common = nb.merge(
    common_keys,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="inner",
)


xgb_common = xgb.merge(
    common_keys,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="inner",
)


# ============================================================
# COMBINE RESULTS
# ============================================================

columns = [
    "setting",
    "horizon_weeks",
    "model_family",
    "model_standard",
    "mae",
    "rmse",
    "mase",
]


all_results = pd.concat(
    [
        baseline_common[
            columns
        ],
        nb_common[
            columns
        ],
        xgb_common[
            columns
        ],
    ],
    ignore_index=True,
)


# ============================================================
# SAVE LONG-FORM RESULTS
# ============================================================

ALL_RESULTS_FILE = (
    TABLE_DIR
    / "all_models_common_comparison.csv"
)


all_results.to_csv(
    ALL_RESULTS_FILE,
    index=False,
)


# ============================================================
# MEDIAN MODEL PERFORMANCE
# ============================================================

summary = (

    all_results

    .groupby(
        [
            "horizon_weeks",
            "model_family",
            "model_standard",
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
    )
)


SUMMARY_FILE = (
    TABLE_DIR
    / "all_models_common_summary.csv"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print(
    "\nALL-MODEL COMMON-COHORT SUMMARY"
)


print(
    summary
    .sort_values(
        [
            "horizon_weeks",
            "median_mase",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# COMPARISON WITH PERSISTENCE
# ============================================================

persistence = (

    all_results[

        all_results[
            "model_standard"
        ]

        == "persistence"
    ]

    [
        [
            "setting",
            "horizon_weeks",
            "mase",
        ]
    ]

    .rename(
        columns={
            "mase":
                "persistence_mase"
        }
    )
)


if persistence.empty:

    raise RuntimeError(
        "Persistence baseline was not found."
    )


relative = all_results.merge(
    persistence,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="left",
)


relative[
    "delta_mase_vs_persistence"
] = (
    relative[
        "mase"
    ]
    -
    relative[
        "persistence_mase"
    ]
)


relative[
    "beats_persistence"
] = (
    relative[
        "delta_mase_vs_persistence"
    ]
    < 0
)


RELATIVE_FILE = (
    TABLE_DIR
    / "all_models_vs_persistence.csv"
)


relative.to_csv(
    RELATIVE_FILE,
    index=False,
)


# ============================================================
# SUMMARY RELATIVE TO PERSISTENCE
# ============================================================

relative_summary = (

    relative

    .groupby(
        [
            "horizon_weeks",
            "model_family",
            "model_standard",
        ],
        as_index=False,
    )

    .agg(

        median_delta_mase_vs_persistence=(
            "delta_mase_vs_persistence",
            "median",
        ),

        settings_beating_persistence=(
            "beats_persistence",
            "sum",
        ),

        settings_evaluated=(
            "setting",
            "nunique",
        ),
    )
)


RELATIVE_SUMMARY_FILE = (
    TABLE_DIR
    / "all_models_vs_persistence_summary.csv"
)


relative_summary.to_csv(
    RELATIVE_SUMMARY_FILE,
    index=False,
)


print(
    "\nPERFORMANCE RELATIVE TO PERSISTENCE"
)


print(
    relative_summary
    .sort_values(
        [
            "horizon_weeks",
            "median_delta_mase_vs_persistence",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FINISH
# ============================================================

print(
    "\nALL-MODEL COMPARISON COMPLETE"
)


print(
    f"Saved: {ALL_RESULTS_FILE}"
)

print(
    f"Saved: {SUMMARY_FILE}"
)

print(
    f"Saved: {RELATIVE_FILE}"
)

print(
    f"Saved: {RELATIVE_SUMMARY_FILE}"
)