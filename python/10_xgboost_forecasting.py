"""
Temporally validated XGBoost dengue forecasting.

Models
------
1. xgboost_epidemiology:
   current/recent dengue history + seasonality

2. xgboost_climate:
   the same epidemiological predictors + climate predictors

Forecast horizons
-----------------
1, 2 and 4 weeks ahead.

Validation design
-----------------
For each setting:
- years before the final two observed years are used for training;
- the penultimate year is used for validation/early stopping;
- the final year is held out as the test set;
- after early stopping, the model is refitted on training + validation data;
- horizon-safe boundaries prevent target leakage across the
  train/validation/test boundaries.

Missing surveillance observations remain missing.
They are never converted to zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


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
        f"Feature dataset not found: {DATA_FILE}"
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
    .reset_index(
        drop=True
    )
)


print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# FEATURES
# ============================================================

EPIDEMIOLOGY_FEATURES = [

    "log_cases",

    "log_cases_lag_1",

    "log_cases_lag_2",

    "log_cases_lag_4",

    "log_cases_lag_8",

    "log_cases_lag_52",

    "log_cases_roll4",

    "season_sin",

    "season_cos",
]


CLIMATE_FEATURES = [

    "log_rain_era5_lag_0",

    "log_rain_era5_lag_1",

    "log_rain_era5_lag_2",

    "log_rain_era5_lag_4",

    "log_rain_era5_lag_6",

    "log_rain_era5_lag_8",


    "temp_era5_lag_0",

    "temp_era5_lag_1",

    "temp_era5_lag_2",

    "temp_era5_lag_4",

    "temp_era5_lag_6",

    "temp_era5_lag_8",


    "humidity_era5_lag_0",

    "humidity_era5_lag_1",

    "humidity_era5_lag_2",

    "humidity_era5_lag_4",

    "humidity_era5_lag_6",

    "humidity_era5_lag_8",
]


MODEL_FEATURES = {

    "xgboost_epidemiology":
        EPIDEMIOLOGY_FEATURES,

    "xgboost_climate":
        EPIDEMIOLOGY_FEATURES
        + CLIMATE_FEATURES,
}


HORIZONS = [
    1,
    2,
    4,
]


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = {

    "setting",

    "analysis_week",

    "source_year",

    "cases",

    *EPIDEMIOLOGY_FEATURES,

    *CLIMATE_FEATURES,
}


for horizon in HORIZONS:

    required_columns.add(
        f"target_log_h{horizon}"
    )

    required_columns.add(
        f"target_cases_h{horizon}"
    )


missing_columns = sorted(
    required_columns.difference(
        df.columns
    )
)


if missing_columns:

    raise KeyError(

        "The feature dataset is missing "
        "required columns:\n"

        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# METRICS
# ============================================================

def mae(
    actual,
    predicted
):

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

    return float(

        np.sqrt(

            np.mean(

                (
                    actual
                    - predicted
                ) ** 2
            )
        )
    )


# ============================================================
# MASE DENOMINATOR
# ============================================================

def mase_scale(
    history
):

    history = (

        history

        .sort_values(
            "analysis_week"
        )

        .copy()
    )


    previous_cases = (

        history[
            "cases"
        ]

        .shift(
            1
        )
    )


    previous_week = (

        history[
            "analysis_week"
        ]

        .shift(
            1
        )
    )


    consecutive = (

        (
            history[
                "analysis_week"
            ]

            - previous_week
        )

        == 1
    )


    valid = (

        history[
            "cases"
        ].notna()

        & previous_cases.notna()

        & consecutive
    )


    if not valid.any():

        return np.nan


    differences = (

        history.loc[
            valid,
            "cases"
        ].to_numpy()

        -

        previous_cases.loc[
            valid
        ].to_numpy()
    )


    return float(

        np.mean(

            np.abs(
                differences
            )
        )
    )


# ============================================================
# XGBOOST TRAINING
# ============================================================

def train_xgboost(

    train_data,

    validation_data,

    features,

    target
):

    """
    First use validation data for early stopping.

    Then refit the final model using training + validation
    data with exactly the selected number of boosting rounds.

    Test data are never used here.
    """


    dtrain = xgb.DMatrix(

        train_data[
            features
        ],

        label=
            train_data[
                target
            ],

        feature_names=
            features,
    )


    dvalidation = xgb.DMatrix(

        validation_data[
            features
        ],

        label=
            validation_data[
                target
            ],

        feature_names=
            features,
    )


    params = {

        "objective":
            "reg:squarederror",

        "eval_metric":
            "rmse",

        "eta":
            0.03,

        "max_depth":
            4,

        "min_child_weight":
            5,

        "subsample":
            0.8,

        "colsample_bytree":
            0.8,

        "lambda":
            1.0,

        "alpha":
            0.0,

        "tree_method":
            "hist",

        "seed":
            42,
    }


    # --------------------------------------------------------
    # STAGE 1
    # Determine optimal number of boosting rounds
    # --------------------------------------------------------

    preliminary_model = xgb.train(

        params=
            params,

        dtrain=
            dtrain,

        num_boost_round=
            2000,

        evals=[
            (
                dvalidation,
                "validation"
            )
        ],

        early_stopping_rounds=
            100,

        verbose_eval=
            False,
    )


    best_rounds = (

        preliminary_model.best_iteration

        + 1
    )


    # --------------------------------------------------------
    # STAGE 2
    # Refit on training + validation
    # --------------------------------------------------------

    final_training = pd.concat(

        [
            train_data,

            validation_data
        ],

        ignore_index=True
    )


    dfinal = xgb.DMatrix(

        final_training[
            features
        ],

        label=
            final_training[
                target
            ],

        feature_names=
            features,
    )


    final_model = xgb.train(

        params=
            params,

        dtrain=
            dfinal,

        num_boost_round=
            best_rounds,

        verbose_eval=
            False,
    )


    return (

        final_model,

        best_rounds
    )


# ============================================================
# TEMPORAL EVALUATION
# ============================================================

results = []


for setting, group in df.groupby(

    "setting",

    sort=True
):


    group = (

        group

        .sort_values(
            "analysis_week"
        )

        .reset_index(
            drop=True
        )

        .copy()
    )


    available_years = sorted(

        group[
            "source_year"
        ]

        .dropna()

        .astype(
            int
        )

        .unique()

        .tolist()
    )


    if len(
        available_years
    ) < 3:

        print(

            f"Skipping {setting}: "
            "fewer than 3 observed years."
        )

        continue


    validation_year = (
        available_years[
            -2
        ]
    )


    test_year = (
        available_years[
            -1
        ]
    )


    validation_positions = np.flatnonzero(

        group[
            "source_year"
        ]

        .eq(
            validation_year
        )

        .to_numpy()
    )


    test_positions = np.flatnonzero(

        group[
            "source_year"
        ]

        .eq(
            test_year
        )

        .to_numpy()
    )


    if (

        len(
            validation_positions
        ) == 0

        or

        len(
            test_positions
        ) == 0
    ):

        print(

            f"Skipping {setting}: "
            "could not locate temporal boundaries."
        )

        continue


    validation_start = int(

        validation_positions[
            0
        ]
    )


    test_start = int(

        test_positions[
            0
        ]
    )


    # --------------------------------------------------------
    # MASE denominator
    #
    # Use all observed history before the test year.
    # --------------------------------------------------------

    scale_data = (

        group

        .iloc[
            :test_start
        ]

        .copy()
    )


    scale = mase_scale(
        scale_data
    )


    print(

        "\n"

        f"{setting}: "

        f"validation={validation_year}, "

        f"test={test_year}"
    )


    positions = np.arange(
        len(
            group
        )
    )


    # ========================================================
    # FORECAST HORIZONS
    # ========================================================

    for horizon in HORIZONS:


        target_log = (

            f"target_log_h{horizon}"
        )


        target_cases = (

            f"target_cases_h{horizon}"
        )


        # ----------------------------------------------------
        # HORIZON-SAFE SPLITTING
        #
        # If origin is t and forecast horizon is h,
        # the target occurs at t+h.
        #
        # Training targets must remain before validation.
        # Validation targets must remain before test.
        # ----------------------------------------------------

        train_mask = (

            positions
            + horizon

            < validation_start
        )


        validation_mask = (

            (
                positions
                >= validation_start
            )

            &

            (
                positions
                + horizon

                < test_start
            )
        )


        test_mask = (

            positions
            >= test_start
        )


        train_full = (

            group.loc[
                train_mask
            ]

            .copy()
        )


        validation_full = (

            group.loc[
                validation_mask
            ]

            .copy()
        )


        test_full = (

            group.loc[
                test_mask
            ]

            .copy()
        )


        # ----------------------------------------------------
        # SAME OBSERVATIONS FOR BOTH MODELS
        #
        # This makes epidemiology-only versus climate
        # comparison fair.
        # ----------------------------------------------------

        complete_case_columns = list(

            dict.fromkeys(

                EPIDEMIOLOGY_FEATURES

                + CLIMATE_FEATURES

                + [

                    target_log,

                    target_cases,
                ]
            )
        )


        train_data = (

            train_full[
                complete_case_columns
            ]

            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

            .dropna()

            .copy()
        )


        validation_data = (

            validation_full[
                complete_case_columns
            ]

            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

            .dropna()

            .copy()
        )


        test_data = (

            test_full[
                complete_case_columns
            ]

            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

            .dropna()

            .copy()
        )


        if (

            len(
                train_data
            ) < 100

            or

            len(
                validation_data
            ) < 10

            or

            len(
                test_data
            ) < 10
        ):

            print(

                f"Skipping {setting}, "

                f"horizon={horizon}: "

                "insufficient complete data "

                f"(train={len(train_data)}, "

                f"validation={len(validation_data)}, "

                f"test={len(test_data)})."
            )

            continue


        # ====================================================
        # MODEL TYPES
        # ====================================================

        for (
            model_name,
            features
        ) in MODEL_FEATURES.items():


            model, best_rounds = train_xgboost(

                train_data=
                    train_data,

                validation_data=
                    validation_data,

                features=
                    features,

                target=
                    target_log,
            )


            dtest = xgb.DMatrix(

                test_data[
                    features
                ],

                feature_names=
                    features,
            )


            prediction_log = model.predict(

                dtest
            )


            predictions = np.expm1(

                prediction_log
            )


            # Negative predicted case counts are impossible.

            predictions = np.clip(

                predictions,

                0,

                None
            )


            actual = (

                test_data[
                    target_cases
                ]

                .to_numpy(
                    dtype=float
                )
            )


            valid_prediction = (

                np.isfinite(
                    actual
                )

                &

                np.isfinite(
                    predictions
                )
            )


            actual_eval = (

                actual[
                    valid_prediction
                ]
            )


            prediction_eval = (

                predictions[
                    valid_prediction
                ]
            )


            if len(
                actual_eval
            ) == 0:

                continue


            model_mae = mae(

                actual_eval,

                prediction_eval
            )


            model_rmse = rmse(

                actual_eval,

                prediction_eval
            )


            if (

                pd.notna(
                    scale
                )

                and

                scale > 0
            ):

                model_mase = (

                    model_mae

                    / scale
                )

            else:

                model_mase = np.nan


            results.append(

                {

                    "setting":
                        setting,

                    "validation_year":
                        validation_year,

                    "test_year":
                        test_year,

                    "horizon_weeks":
                        horizon,

                    "model":
                        model_name,

                    "n_train":
                        len(
                            train_data
                        ),

                    "n_validation":
                        len(
                            validation_data
                        ),

                    "n_test":
                        len(
                            actual_eval
                        ),

                    "best_rounds":
                        best_rounds,

                    "mae":
                        model_mae,

                    "rmse":
                        model_rmse,

                    "mase":
                        model_mase,
                }
            )


# ============================================================
# FULL RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(

        "No XGBoost models were "
        "successfully evaluated."
    )


RESULTS_FILE = (

    OUTPUT_DIR

    / "xgboost_forecasting_results.csv"
)


results_df.to_csv(

    RESULTS_FILE,

    index=False
)


print(

    "\nXGBOOST FORECASTING RESULTS"
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


SUMMARY_FILE = (

    OUTPUT_DIR

    / "xgboost_forecasting_summary.csv"
)


summary.to_csv(

    SUMMARY_FILE,

    index=False
)


print(

    "\nMEDIAN XGBOOST PERFORMANCE "
    "ACROSS SETTINGS"
)


print(

    summary.to_string(
        index=False
    )
)


# ============================================================
# CLIMATE ABLATION
# ============================================================

epi = (

    results_df[

        results_df[
            "model"
        ]

        == "xgboost_epidemiology"
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
                "mase_epidemiology"
        }
    )
)


climate = (

    results_df[

        results_df[
            "model"
        ]

        == "xgboost_climate"
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
                "mase_climate"
        }
    )
)


ablation = epi.merge(

    climate,

    on=[
        "setting",

        "horizon_weeks",
    ],

    how="inner"
)


ablation[

    "delta_mase_climate_minus_epi"

] = (

    ablation[
        "mase_climate"
    ]

    -

    ablation[
        "mase_epidemiology"
    ]
)


ablation[

    "climate_improves"

] = (

    ablation[

        "delta_mase_climate_minus_epi"

    ]

    < 0
)


ABLATION_FILE = (

    OUTPUT_DIR

    / "xgboost_climate_ablation.csv"
)


ablation.to_csv(

    ABLATION_FILE,

    index=False
)


print(

    "\nXGBOOST CLIMATE ABLATION"
)


print(

    ablation.to_string(
        index=False
    )
)


# ============================================================
# CLIMATE ABLATION SUMMARY
# ============================================================

ablation_summary = (

    ablation

    .groupby(

        "horizon_weeks",

        as_index=False
    )

    .agg(

        settings_evaluated=(
            "setting",
            "nunique"
        ),

        settings_climate_improves=(
            "climate_improves",
            "sum"
        ),

        median_delta_mase=(
            "delta_mase_climate_minus_epi",
            "median"
        ),
    )
)


ABLATION_SUMMARY_FILE = (

    OUTPUT_DIR

    / "xgboost_climate_ablation_summary.csv"
)


ablation_summary.to_csv(

    ABLATION_SUMMARY_FILE,

    index=False
)


print(

    "\nXGBOOST CLIMATE ABLATION SUMMARY"
)


print(

    ablation_summary.to_string(
        index=False
    )
)


# ============================================================
# FINISH
# ============================================================

print(

    "\nXGBOOST FORECASTING COMPLETE"
)


print(

    f"Saved: {RESULTS_FILE}"
)


print(

    f"Saved: {SUMMARY_FILE}"
)


print(

    f"Saved: {ABLATION_FILE}"
)


print(

    f"Saved: {ABLATION_SUMMARY_FILE}"
)