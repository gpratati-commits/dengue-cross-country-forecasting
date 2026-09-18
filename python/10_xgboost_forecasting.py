"""
Leakage-safe multi-horizon dengue forecasting with XGBoost.

Models
------
1. xgboost_epidemiology
   Dengue-history + seasonal predictors.

2. xgboost_climate
   Dengue-history + seasonal predictors + climate predictors.

Horizons
--------
1, 2 and 4 weeks ahead.

Leakage protection
------------------
Train/validation/test membership is determined by the YEAR OF THE
FORECAST TARGET, not the year of the forecast origin.

Therefore, a forecast made late in one calendar year for an outcome
in the next calendar year is assigned to the correct target period.

Outputs
-------
- outputs/tables/xgboost_forecasting_results.csv
- outputs/tables/xgboost_forecasting_summary.csv
- outputs/tables/xgboost_climate_ablation.csv
- outputs/tables/xgboost_climate_ablation_summary.csv
- outputs/predictions/xgboost_predictions.csv
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


TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
        ]
        .to_numpy(
            dtype=float
        )
        - previous_cases.loc[
            valid
        ]
        .to_numpy(
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
# EPIDEMIOLOGICAL FEATURES
# ============================================================

def add_safe_epidemiological_features(
    data
):
    """
    Create predictors using information available at
    the forecast origin.
    """

    data = (
        data
        .sort_values(
            [
                "setting",
                "analysis_week",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    data[
        "log_cases"
    ] = np.log1p(
        data[
            "cases"
        ].clip(
            lower=0
        )
    )

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

    grouped_log_cases = (
        data
        .groupby(
            "setting",
            sort=False
        )[
            "log_cases"
        ]
    )

    for lag in [
        1,
        2,
        4,
        8,
    ]:

        data[
            f"log_cases_lag_{lag}"
        ] = grouped_log_cases.shift(
            lag
        )

    for window in [
        4,
        8,
    ]:

        data[
            f"log_cases_roll_mean_{window}"
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
    Automatically detect rainfall, temperature,
    and humidity columns.
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

    return {
        name: column
        for (
            name,
            column
        ) in climate.items()
        if column is not None
    }


def add_climate_features(
    data,
    climate_columns
):
    """
    Create current-origin and lagged climate predictors.
    """

    data = data.copy()

    climate_features = []

    for (
        climate_name,
        source_column
    ) in climate_columns.items():

        data[
            source_column
        ] = pd.to_numeric(
            data[
                source_column
            ],
            errors="coerce"
        )

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

def add_targets(
    data
):
    """
    Create future outcomes and true target-time metadata.

    The continuity check ensures that row-based shifting
    represents an actual h-week forecast.
    """

    data = data.copy()

    for horizon in HORIZONS:

        grouped = data.groupby(
            "setting",
            sort=False
        )

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

        data[
            target_cases
        ] = (
            grouped[
                "cases"
            ]
            .shift(
                -horizon
            )
        )

        data[
            target_year
        ] = (
            grouped[
                "source_year"
            ]
            .shift(
                -horizon
            )
        )

        data[
            target_week
        ] = (
            grouped[
                "source_week"
            ]
            .shift(
                -horizon
            )
        )

        data[
            target_analysis_week
        ] = (
            grouped[
                "analysis_week"
            ]
            .shift(
                -horizon
            )
        )

        expected_target_week = (
            data[
                "analysis_week"
            ]
            + horizon
        )

        continuous = (
            data[
                target_analysis_week
            ]
            .eq(
                expected_target_week
            )
        )

        for column in [
            target_cases,
            target_year,
            target_week,
            target_analysis_week,
        ]:

            data.loc[
                ~continuous,
                column
            ] = np.nan

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
    Select boosting rounds using the validation
    target year.

    Then refit using training + validation data.
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

    best_iteration = getattr(
        preliminary_model,
        "best_iteration",
        None
    )

    if best_iteration is None:

        best_rounds = (
            MAX_BOOST_ROUNDS
        )

    else:

        best_rounds = (
            int(
                best_iteration
            )
            + 1
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
    Generate non-negative predictions on
    the original case-count scale.
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
        "Processed feature file not found: "
        f"{DATA_FILE}"
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

required_columns = {
    "setting",
    "source_year",
    "source_week",
    "cases",
}


missing_required = sorted(
    required_columns
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
# CLEAN CORE VARIABLES
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


data = (
    data
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


# ============================================================
# BUILD EPIDEMIOLOGICAL FEATURES
# ============================================================

data = (
    add_safe_epidemiological_features(
        data
    )
)


# ============================================================
# BUILD CLIMATE FEATURES
# ============================================================

climate_columns = (
    choose_climate_columns(
        data
    )
)


if not climate_columns:

    raise RuntimeError(
        "No climate predictors were detected. "
        "Check the climate column names in "
        "data/processed/dengue_model_features.csv."
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
) = add_climate_features(
    data,
    climate_columns
)


# ============================================================
# BUILD LEAKAGE-SAFE TARGETS
# ============================================================

data = add_targets(
    data
)


# ============================================================
# MODEL FEATURE SETS
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
    for column in epi_features
    if column
    in data.columns
]


if not epi_features:

    raise RuntimeError(
        "No epidemiological XGBoost "
        "features could be constructed."
    )


climate_model_features = list(
    dict.fromkeys(
        epi_features
        + climate_features
    )
)


print(
    "\nEpidemiology features:",
    epi_features
)


print(
    "\nClimate source columns:",
    climate_columns
)


print(
    "\nClimate model features:",
    climate_features
)


# ============================================================
# RESULT STORAGE
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
# TEMPORALLY VALIDATED MODEL LOOP
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
            "fewer than three observed years."
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
        f"validation target year="
        f"{validation_year}, "
        f"test target year="
        f"{test_year}"
    )


    # --------------------------------------------------------
    # MASE SCALE
    #
    # Use observed dengue history before the test target year.
    # --------------------------------------------------------

    scale_training = (
        group.loc[
            group[
                "source_year"
            ]
            < test_year
        ]
        .copy()
    )


    scale = mase_scale(
        scale_training
    )


    for horizon in HORIZONS:

        target_cases_col = (
            f"target_cases_h{horizon}"
        )

        target_year_col = (
            f"target_year_h{horizon}"
        )

        target_week_col = (
            f"target_week_h{horizon}"
        )

        target_analysis_week_col = (
            f"target_analysis_week_h{horizon}"
        )


        # ----------------------------------------------------
        # USE A COMMON COMPLETE-DATA COHORT
        #
        # Both XGBoost variants therefore use identical rows.
        # ----------------------------------------------------

        common_required = list(
            dict.fromkeys(
                climate_model_features
                + [
                    target_cases_col,
                    target_year_col,
                    target_week_col,
                    target_analysis_week_col,
                ]
            )
        )


        complete = (
            group.loc[
                group[
                    common_required
                ]
                .notna()
                .all(
                    axis=1
                )
            ]
            .copy()
        )


        complete[
            climate_model_features
        ] = (
            complete[
                climate_model_features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan
            )
        )


        complete = (
            complete
            .dropna(
                subset=
                    climate_model_features
            )
            .copy()
        )


        # ====================================================
        # LEAKAGE-SAFE TEMPORAL SPLIT
        #
        # IMPORTANT:
        # SPLIT ON TARGET YEAR, NOT SOURCE YEAR.
        # ====================================================

        train_data = (
            complete.loc[
                complete[
                    target_year_col
                ]
                < validation_year
            ]
            .copy()
        )


        validation_data = (
            complete.loc[
                complete[
                    target_year_col
                ]
                == validation_year
            ]
            .copy()
        )


        test_data = (
            complete.loc[
                complete[
                    target_year_col
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
                "insufficient complete "
                "leakage-safe data "
                f"(train={len(train_data)}, "
                f"validation="
                f"{len(validation_data)}, "
                f"test={len(test_data)})."
            )

            continue


        # ----------------------------------------------------
        # DEFENSIVE LEAKAGE CHECKS
        # ----------------------------------------------------

        if not (
            train_data[
                target_year_col
            ]
            < validation_year
        ).all():

            raise RuntimeError(
                f"Training leakage detected "
                f"for {setting}, "
                f"horizon={horizon}."
            )


        if not (
            validation_data[
                target_year_col
            ]
            == validation_year
        ).all():

            raise RuntimeError(
                f"Validation-period leakage "
                f"detected for {setting}, "
                f"horizon={horizon}."
            )


        if not (
            test_data[
                target_year_col
            ]
            == test_year
        ).all():

            raise RuntimeError(
                f"Test-period leakage detected "
                f"for {setting}, "
                f"horizon={horizon}."
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
                    target_cases_col,
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
                    target_cases_col
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


            if not valid_prediction.any():

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
            # PERFORMANCE METRICS
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
            # STORE PREDICTION-LEVEL RESULT
            # ------------------------------------------------

            prediction_detail = pd.DataFrame(
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

                    "target_year":
                        evaluation_rows[
                            target_year_col
                        ]
                        .astype(int)
                        .to_numpy(),

                    "target_week":
                        evaluation_rows[
                            target_week_col
                        ]
                        .astype(int)
                        .to_numpy(),

                    "target_analysis_week":
                        evaluation_rows[
                            target_analysis_week_col
                        ]
                        .astype(int)
                        .to_numpy(),

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


            prediction_frames.append(
                prediction_detail
            )


# ============================================================
# BUILD FULL RESULTS TABLE
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


# ============================================================
# BUILD PREDICTION-LEVEL TABLE
# ============================================================

if not prediction_frames:

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
            "target_analysis_week",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PREDICTION SAFETY CHECKS
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
        subset=
            duplicate_key
    )
    .sum()
)


if (
    duplicate_count
    > 0
):

    raise RuntimeError(
        "Duplicate XGBoost forecast "
        "rows detected: "
        f"{duplicate_count}"
    )


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


if (
    wrong_target_year
    > 0
):

    raise RuntimeError(
        "Leakage-safe XGBoost "
        "test-cohort check failed: "
        f"{wrong_target_year} rows "
        "have target_year != test_year."
    )


missing_actual = int(
    predictions_df[
        "actual"
    ]
    .isna()
    .sum()
)


missing_predicted = int(
    predictions_df[
        "predicted"
    ]
    .isna()
    .sum()
)


if (
    missing_actual
    > 0
    or missing_predicted
    > 0
):

    raise RuntimeError(
        "Missing values remain in "
        "XGBoost prediction output."
    )


cross_year_test_forecasts = int(
    (
        predictions_df[
            "source_year"
        ]
        != predictions_df[
            "target_year"
        ]
    )
    .sum()
)


# ============================================================
# SAVE FULL RESULTS
# ============================================================

results_file = (
    TABLE_DIR
    / "xgboost_forecasting_results.csv"
)


prediction_file = (
    PREDICTION_DIR
    / "xgboost_predictions.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


predictions_df.to_csv(
    prediction_file,
    index=False
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
    "\nMEDIAN PERFORMANCE ACROSS SETTINGS"
)


print(
    summary.to_string(
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


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

print(
    "\nXGBOOST PREDICTION DIAGNOSTICS"
)


print(
    "Rows:",
    len(
        predictions_df
    )
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
    missing_actual
)


print(
    "Missing predicted:",
    missing_predicted
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
    "Cross-year test forecasts retained:",
    cross_year_test_forecasts
)


# ============================================================
# COMPLETE
# ============================================================

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