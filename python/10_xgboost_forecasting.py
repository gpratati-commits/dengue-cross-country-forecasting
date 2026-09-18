"""
Temporally validated multi-horizon dengue forecasting with XGBoost.

Two XGBoost models are evaluated:

1. xgboost_epidemiology
   Dengue-history + seasonal predictors.

2. xgboost_climate
   Dengue-history + seasonal predictors + climate predictors.

Temporal evaluation:
- Training = years before validation year
- Validation = penultimate observed year
- Test = latest observed year

The validation year selects the number of boosting rounds using
early stopping. The model is then refitted using training +
validation data and evaluated on the held-out test year.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import xgboost as xgb


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

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HORIZONS = [
    1,
    2,
    4,
]


SEED = 42

MIN_TRAIN_ROWS = 52
MIN_VALIDATION_ROWS = 20
MIN_TEST_ROWS = 20

MAX_BOOST_ROUNDS = 2000

EARLY_STOPPING_ROUNDS = 100


XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "seed": SEED,
    "tree_method": "hist",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_existing(
    columns,
    candidates
):
    """
    Return the first candidate column that exists.
    """

    for column in candidates:

        if column in columns:

            return column

    return None


def mae(
    actual,
    predicted
):
    """
    Mean absolute error.
    """

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
    """
    Root mean squared error.
    """

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
    Calculate the one-step naive scaling term for MASE.

    Only consecutive observed weeks are used.
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

    consecutive = (
        training[
            "analysis_week"
        ]
        - previous_week
    ).eq(1)

    valid = (
        training[
            "cases"
        ].notna()
        & previous_cases.notna()
        & consecutive
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
# FEATURE ENGINEERING
# ============================================================

def add_safe_features(
    data
):
    """
    Create leakage-safe epidemiological predictors.
    """

    data = (
        data
        .copy()
        .sort_values(
            [
                "setting",
                "analysis_week",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        "log_cases"
        not in data.columns
    ):

        data[
            "log_cases"
        ] = np.log1p(
            data[
                "cases"
            ].clip(
                lower=0
            )
        )

    if (
        "season_sin"
        not in data.columns
    ):

        data[
            "season_sin"
        ] = np.sin(
            2.0
            * math.pi
            * data[
                "source_week"
            ]
            / 52.0
        )

    if (
        "season_cos"
        not in data.columns
    ):

        data[
            "season_cos"
        ] = np.cos(
            2.0
            * math.pi
            * data[
                "source_week"
            ]
            / 52.0
        )

    for lag in [
        1,
        2,
        4,
        8,
    ]:

        column = (
            f"log_cases_lag_{lag}"
        )

        if (
            column
            not in data.columns
        ):

            data[
                column
            ] = (
                data
                .groupby(
                    "setting",
                    sort=False
                )[
                    "log_cases"
                ]
                .shift(
                    lag
                )
            )

    for window in [
        4,
        8,
    ]:

        column = (
            f"log_cases_roll_mean_{window}"
        )

        if (
            column
            not in data.columns
        ):

            data[
                column
            ] = (
                data
                .groupby(
                    "setting",
                    sort=False
                )[
                    "log_cases"
                ]
                .transform(
                    lambda series: (
                        series
                        .shift(1)
                        .rolling(
                            window=window,
                            min_periods=window
                        )
                        .mean()
                    )
                )
            )

    return data


# ============================================================
# CLIMATE VARIABLES
# ============================================================

def choose_climate_columns(
    data
):
    """
    Detect rainfall, temperature and humidity columns.
    """

    columns = set(
        data.columns
    )

    rainfall = first_existing(
        columns,
        [
            "log_rain_era5",
            "rain_era5",
            "rainfall_era5",
            "rainfall",
            "Rainfall",
            "precipitation",
            "Precipitation",
        ]
    )

    temperature = first_existing(
        columns,
        [
            "temp_era5",
            "temperature_era5",
            "temperature",
            "Temperature",
            "TempMean_MERRA2",
            "Temp_MERRA2",
        ]
    )

    humidity = first_existing(
        columns,
        [
            "humidity_era5",
            "relative_humidity_era5",
            "humidity",
            "Humidity",
            "RH",
        ]
    )

    climate = {
        "rainfall":
            rainfall,

        "temperature":
            temperature,

        "humidity":
            humidity,
    }

    climate = {
        name: column
        for (
            name,
            column
        )
        in climate.items()
        if column is not None
    }

    return climate


def add_climate_lags(
    data,
    climate_columns
):
    """
    Create current-source-week and lagged climate predictors.
    """

    data = data.copy()

    climate_features = []

    for (
        climate_name,
        source_column
    ) in climate_columns.items():

        current_column = (
            f"climate_{climate_name}_lag_0"
        )

        data[
            current_column
        ] = data[
            source_column
        ]

        climate_features.append(
            current_column
        )

        for lag in [
            1,
            2,
            4,
            6,
            8,
        ]:

            lag_column = (
                f"climate_{climate_name}_lag_{lag}"
            )

            data[
                lag_column
            ] = (
                data
                .groupby(
                    "setting",
                    sort=False
                )[
                    source_column
                ]
                .shift(
                    lag
                )
            )

            climate_features.append(
                lag_column
            )

    return (
        data,
        climate_features
    )


# ============================================================
# FORECAST TARGETS
# ============================================================

def add_targets_if_needed(
    data
):
    """
    Create future case targets only when they do not already exist.
    """

    data = data.copy()

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
                    "setting",
                    sort=False
                )[
                    "cases"
                ]
                .shift(
                    -horizon
                )
            )

    return data


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
    Select boosting rounds using validation data.

    Then refit on training + validation data.
    """

    dtrain = xgb.DMatrix(
        train_data[
            features
        ],
        label=np.log1p(
            train_data[
                target
            ].clip(
                lower=0
            )
        ),
        feature_names=features,
    )

    dvalidation = xgb.DMatrix(
        validation_data[
            features
        ],
        label=np.log1p(
            validation_data[
                target
            ].clip(
                lower=0
            )
        ),
        feature_names=features,
    )

    preliminary_model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=
            MAX_BOOST_ROUNDS,
        evals=[
            (
                dvalidation,
                "validation"
            )
        ],
        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    if hasattr(
        preliminary_model,
        "best_iteration"
    ):

        best_rounds = (
            int(
                preliminary_model
                .best_iteration
            )
            + 1
        )

    else:

        best_rounds = (
            MAX_BOOST_ROUNDS
        )

    best_rounds = max(
        1,
        min(
            best_rounds,
            MAX_BOOST_ROUNDS
        )
    )

    final_training = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    dfinal = xgb.DMatrix(
        final_training[
            features
        ],
        label=np.log1p(
            final_training[
                target
            ].clip(
                lower=0
            )
        ),
        feature_names=features,
    )

    final_model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dfinal,
        num_boost_round=
            best_rounds,
        verbose_eval=False,
    )

    return (
        final_model,
        best_rounds
    )


def predict_cases(
    model,
    data,
    features
):
    """
    Generate case-count predictions.
    """

    dtest = xgb.DMatrix(
        data[
            features
        ],
        feature_names=features,
    )

    prediction_log = (
        model.predict(
            dtest
        )
    )

    prediction = np.expm1(
        prediction_log
    )

    prediction = np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )

    return prediction


# ============================================================
# READ DATA
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "Processed feature file "
        f"not found: {DATA_FILE}"
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


missing_required = sorted(
    required
    - set(
        data.columns
    )
)


if missing_required:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(
            missing_required
        )
    )


# ============================================================
# STANDARDISE CORE VARIABLES
# ============================================================

data[
    "source_year"
] = pd.to_numeric(
    data[
        "source_year"
    ],
    errors="coerce"
)


data[
    "source_week"
] = pd.to_numeric(
    data[
        "source_week"
    ],
    errors="coerce"
)


data[
    "cases"
] = pd.to_numeric(
    data[
        "cases"
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

    data = (
        data
        .sort_values(
            [
                "setting",
                "source_year",
                "source_week",
            ]
        )
        .copy()
    )

    data[
        "analysis_week"
    ] = (
        data
        .groupby(
            "setting",
            sort=False
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
# BUILD FEATURES
# ============================================================

data = add_safe_features(
    data
)


climate_columns = (
    choose_climate_columns(
        data
    )
)


if (
    len(
        climate_columns
    )
    < 3
):

    print(
        "\nWarning: fewer than three "
        "climate variable families "
        "were detected."
    )

    print(
        "Detected climate columns:",
        climate_columns
    )


(
    data,
    climate_features
) = add_climate_lags(
    data,
    climate_columns
)


data = add_targets_if_needed(
    data
)


# ============================================================
# EPIDEMIOLOGICAL FEATURES
# ============================================================

epi_features = [
    "log_cases",
    "log_cases_lag_1",
    "log_cases_lag_2",
    "log_cases_lag_4",
    "log_cases_lag_8",
    "log_cases_roll_mean_4",
    "log_cases_roll_mean_8",
    "season_sin",
    "season_cos",
]


epi_features = [
    column
    for column
    in epi_features
    if column
    in data.columns
]


if (
    len(
        epi_features
    )
    == 0
):

    raise RuntimeError(
        "No epidemiological "
        "XGBoost features "
        "could be constructed."
    )


climate_model_features = (
    epi_features
    + climate_features
)


climate_model_features = list(
    dict.fromkeys(
        climate_model_features
    )
)


if (
    len(
        climate_features
    )
    == 0
):

    raise RuntimeError(
        "No climate predictors were detected. "
        "Check the climate column names in "
        "data/processed/"
        "dengue_model_features.csv."
    )


print(
    "\nEpidemiology features:",
    epi_features
)


print(
    "\nClimate features:",
    climate_features
)


# ============================================================
# MODEL STORAGE
# ============================================================

results = []

prediction_frames = []


settings = sorted(
    data[
        "setting"
    ]
    .dropna()
    .unique()
)


# ============================================================
# MODEL LOOP
# ============================================================

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
        < 3
    ):

        print(
            f"\nSkipping {setting}: "
            "fewer than three "
            "observed years."
        )

        continue

    validation_year = (
        observed_years[
            -2
        ]
    )

    test_year = (
        observed_years[
            -1
        ]
    )

    print(
        f"\n{setting}: "
        f"validation={validation_year}, "
        f"test={test_year}"
    )

    for horizon in HORIZONS:

        target = (
            f"target_cases_h{horizon}"
        )

        if (
            target
            not in group.columns
        ):

            print(
                f"Skipping {setting}, "
                f"horizon={horizon}: "
                f"{target} is missing."
            )

            continue


        # ----------------------------------------------------
        # COMMON COMPLETE-DATA COHORT
        # ----------------------------------------------------

        common_required = list(
            dict.fromkeys(
                climate_model_features
                + [
                    target
                ]
            )
        )

        complete_mask = (
            group[
                common_required
            ]
            .notna()
            .all(
                axis=1
            )
        )

        complete = (
            group.loc[
                complete_mask
            ]
            .copy()
        )


        train_data = (
            complete.loc[
                complete[
                    "source_year"
                ]
                < validation_year
            ]
            .copy()
        )


        validation_data = (
            complete.loc[
                complete[
                    "source_year"
                ]
                == validation_year
            ]
            .copy()
        )


        test_data = (
            complete.loc[
                complete[
                    "source_year"
                ]
                == test_year
            ]
            .copy()
        )


        if (
            len(
                train_data
            )
            < MIN_TRAIN_ROWS
            or len(
                validation_data
            )
            < MIN_VALIDATION_ROWS
            or len(
                test_data
            )
            < MIN_TEST_ROWS
        ):

            print(
                f"Skipping {setting}, "
                f"horizon={horizon}: "
                "insufficient complete data "
                f"(train={len(train_data)}, "
                f"validation="
                f"{len(validation_data)}, "
                f"test={len(test_data)})."
            )

            continue


        # ----------------------------------------------------
        # MASE SCALE
        # ----------------------------------------------------

        scale_training = (
            group.loc[
                group[
                    "source_year"
                ]
                < validation_year
            ]
            .copy()
        )

        scale = mase_scale(
            scale_training
        )


        # ----------------------------------------------------
        # TWO XGBOOST MODELS
        # ----------------------------------------------------

        models = {
            "xgboost_epidemiology":
                epi_features,

            "xgboost_climate":
                climate_model_features,
        }


        for (
            model_name,
            features
        ) in models.items():

            (
                model,
                best_rounds
            ) = train_xgboost(
                train_data=
                    train_data,

                validation_data=
                    validation_data,

                features=
                    features,

                target=
                    target,
            )


            prediction_eval = (
                predict_cases(
                    model=
                        model,

                    data=
                        test_data,

                    features=
                        features,
                )
            )


            actual_eval = (
                test_data[
                    target
                ]
                .to_numpy(
                    dtype=float
                )
            )


            valid_prediction = (
                np.isfinite(
                    actual_eval
                )
                & np.isfinite(
                    prediction_eval
                )
            )


            if (
                not valid_prediction.any()
            ):

                print(
                    f"Skipping {setting}, "
                    f"horizon={horizon}, "
                    f"model={model_name}: "
                    "no finite predictions."
                )

                continue


            actual_eval = (
                actual_eval[
                    valid_prediction
                ]
            )


            prediction_eval = (
                prediction_eval[
                    valid_prediction
                ]
            )


            evaluation_rows = (
                test_data
                .iloc[
                    np.flatnonzero(
                        valid_prediction
                    )
                ]
                .copy()
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # STORE SUMMARY RESULT
            # ------------------------------------------------

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
                            evaluation_rows
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


            # ------------------------------------------------
            # STORE INDIVIDUAL PREDICTIONS
            # ------------------------------------------------

            prediction_detail = (
                pd.DataFrame(
                    {
                        "setting":
                            setting,

                        "validation_year":
                            validation_year,

                        "test_year":
                            test_year,

                        "source_year":
                            evaluation_rows[
                                "source_year"
                            ]
                            .to_numpy(),

                        "source_week":
                            evaluation_rows[
                                "source_week"
                            ]
                            .to_numpy(),

                        "week_label":
                            evaluation_rows[
                                "week_label"
                            ]
                            .astype(str)
                            .to_numpy(),

                        "analysis_week":
                            evaluation_rows[
                                "analysis_week"
                            ]
                            .to_numpy(),

                        "target_analysis_week":
                            (
                                evaluation_rows[
                                    "analysis_week"
                                ]
                                .to_numpy()
                                + horizon
                            ),

                        "horizon_weeks":
                            horizon,

                        "model":
                            model_name,

                        "actual":
                            actual_eval,

                        "predicted":
                            prediction_eval,
                    }
                )
            )


            prediction_frames.append(
                prediction_detail
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


results_file = (
    TABLE_DIR
    / "xgboost_forecasting_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE PREDICTION-LEVEL RESULTS
# ============================================================

if (
    len(
        prediction_frames
    )
    == 0
):

    raise RuntimeError(
        "No XGBoost prediction-level "
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


prediction_file = (
    PREDICTION_DIR
    / "xgboost_predictions.csv"
)


predictions_df.to_csv(
    prediction_file,
    index=False
)


print(
    "\nSaved prediction-level data:",
    prediction_file
)


# ============================================================
# MEDIAN PERFORMANCE ACROSS SETTINGS
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


summary_file = (
    TABLE_DIR
    / "xgboost_forecasting_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# CLIMATE ABLATION
# ============================================================

ablation_source = (
    results_df[
        [
            "setting",
            "horizon_weeks",
            "model",
            "mase",
        ]
    ]
    .copy()
)


ablation_wide = (
    ablation_source
    .pivot_table(
        index=[
            "setting",
            "horizon_weeks",
        ],

        columns=
            "model",

        values=
            "mase",

        aggfunc=
            "first",
    )
    .reset_index()
)


needed_ablation_columns = {
    "xgboost_epidemiology",
    "xgboost_climate",
}


if (
    needed_ablation_columns
    .issubset(
        ablation_wide.columns
    )
):

    ablation = (
        ablation_wide[
            [
                "setting",
                "horizon_weeks",
                "xgboost_epidemiology",
                "xgboost_climate",
            ]
        ]
        .copy()
    )


    ablation = (
        ablation
        .rename(
            columns={
                "xgboost_epidemiology":
                    "mase_epidemiology",

                "xgboost_climate":
                    "mase_climate",
            }
        )
    )


    ablation[
        "delta_mase_climate_minus_epi"
    ] = (
        ablation[
            "mase_climate"
        ]
        - ablation[
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


else:

    ablation = pd.DataFrame(
        columns=[
            "setting",
            "horizon_weeks",
            "mase_epidemiology",
            "mase_climate",
            "delta_mase_climate_minus_epi",
            "climate_improves",
        ]
    )


ablation_file = (
    TABLE_DIR
    / "xgboost_climate_ablation.csv"
)


ablation.to_csv(
    ablation_file,
    index=False
)


# ============================================================
# CLIMATE ABLATION SUMMARY
# ============================================================

if not ablation.empty:

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


else:

    ablation_summary = (
        pd.DataFrame(
            columns=[
                "horizon_weeks",
                "settings_evaluated",
                "settings_climate_improves",
                "median_delta_mase",
            ]
        )
    )


ablation_summary_file = (
    TABLE_DIR
    / "xgboost_climate_ablation_summary.csv"
)


ablation_summary.to_csv(
    ablation_summary_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\nXGBOOST FORECASTING RESULTS"
)


print(
    results_df.to_string(
        index=False
    )
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
    "\nXGBOOST CLIMATE ABLATION"
)


if ablation.empty:

    print(
        "No paired climate/epidemiology "
        "XGBoost comparisons were available."
    )


else:

    print(
        ablation.to_string(
            index=False
        )
    )


print(
    "\nXGBOOST CLIMATE ABLATION SUMMARY"
)


print(
    ablation_summary.to_string(
        index=False
    )
)


print(
    "\nXGBOOST FORECASTING COMPLETE"
)


print(
    f"Saved: {results_file}"
)


print(
    f"Saved: {summary_file}"
)


print(
    f"Saved: {ablation_file}"
)


print(
    f"Saved: {ablation_summary_file}"
)


print(
    f"Saved: {prediction_file}"
)