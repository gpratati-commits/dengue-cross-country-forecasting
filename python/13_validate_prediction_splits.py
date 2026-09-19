"""
Independent validation of prediction-level outputs.

Checks:
1. Required prediction files exist.
2. Required columns exist.
3. No missing actual or predicted values.
4. No non-finite actual or predicted values.
5. No duplicate forecast rows.
6. target_year == test_year for every exported test prediction.
7. target_analysis_week == analysis_week + horizon_weeks.
8. Forecast horizons are 1, 2 and 4 weeks.
9. Expected model names are present.
10. Reports setting coverage and legitimate cross-year forecasts.

This script does NOT require every forecasting family to cover every
setting. Coverage differences are reported explicitly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PREDICTION_DIR = Path("outputs/predictions")

TABLE_DIR = Path("outputs/tables")

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {
    "baseline": (
        PREDICTION_DIR
        / "baseline_predictions.csv"
    ),
    "negative_binomial": (
        PREDICTION_DIR
        / "negative_binomial_predictions.csv"
    ),
    "xgboost": (
        PREDICTION_DIR
        / "xgboost_predictions.csv"
    ),
}


EXPECTED_MODELS = {
    "baseline": {
        "persistence",
        "seasonal_naive",
        "moving_average_4",
    },
    "negative_binomial": {
        "epidemiology_only",
        "climate_informed",
    },
    "xgboost": {
        "xgboost_epidemiology",
        "xgboost_climate",
    },
}


EXPECTED_HORIZONS = {
    1,
    2,
    4,
}


REQUIRED_COLUMNS = {
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
}


# ============================================================
# STORAGE
# ============================================================

validation_rows = []

overall_pass = True


# ============================================================
# VALIDATE EACH FORECAST FAMILY
# ============================================================

for family, file_path in FILES.items():

    print(
        "\n"
        + "=" * 70
    )

    print(
        family.upper()
    )

    print(
        file_path
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # FILE EXISTS
    # --------------------------------------------------------

    if not file_path.exists():

        print(
            "FAIL: prediction file does not exist."
        )

        overall_pass = False

        validation_rows.append(
            {
                "family": family,
                "rows": 0,
                "settings": 0,
                "models": "",
                "horizons": "",
                "missing_actual": np.nan,
                "missing_predicted": np.nan,
                "nonfinite_actual": np.nan,
                "nonfinite_predicted": np.nan,
                "duplicates": np.nan,
                "wrong_target_year": np.nan,
                "wrong_target_analysis_week": np.nan,
                "cross_year_forecasts": np.nan,
                "expected_models_present": False,
                "expected_horizons_present": False,
                "validation_passed": False,
            }
        )

        continue


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path
    )


    print(
        f"Rows: {len(df):,}"
    )


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = sorted(
        REQUIRED_COLUMNS
        - set(
            df.columns
        )
    )


    if missing_columns:

        print(
            "FAIL: missing required columns:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        overall_pass = False

        validation_rows.append(
            {
                "family": family,
                "rows": len(df),
                "settings": (
                    df["setting"].nunique()
                    if "setting" in df.columns
                    else np.nan
                ),
                "models": "",
                "horizons": "",
                "missing_actual": np.nan,
                "missing_predicted": np.nan,
                "nonfinite_actual": np.nan,
                "nonfinite_predicted": np.nan,
                "duplicates": np.nan,
                "wrong_target_year": np.nan,
                "wrong_target_analysis_week": np.nan,
                "cross_year_forecasts": np.nan,
                "expected_models_present": False,
                "expected_horizons_present": False,
                "validation_passed": False,
            }
        )

        continue


    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    settings = sorted(
        df[
            "setting"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    models = sorted(
        df[
            "model"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    horizons = sorted(
        df[
            "horizon_weeks"
        ]
        .dropna()
        .astype(int)
        .unique()
    )


    print(
        "Settings:",
        settings,
    )

    print(
        "Horizons:",
        horizons,
    )

    print(
        "Models:",
        models,
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_actual = int(
        df[
            "actual"
        ]
        .isna()
        .sum()
    )


    missing_predicted = int(
        df[
            "predicted"
        ]
        .isna()
        .sum()
    )


    # --------------------------------------------------------
    # NON-FINITE VALUES
    # --------------------------------------------------------

    actual_numeric = df[
        "actual"
    ].to_numpy(
        dtype=float
    )


    predicted_numeric = df[
        "predicted"
    ].to_numpy(
        dtype=float
    )


    nonfinite_actual = int(
        (
            ~np.isfinite(
                actual_numeric
            )
        )
        .sum()
    )


    nonfinite_predicted = int(
        (
            ~np.isfinite(
                predicted_numeric
            )
        )
        .sum()
    )


    # --------------------------------------------------------
    # DUPLICATE FORECASTS
    # --------------------------------------------------------

    duplicate_columns = [
        "setting",
        "analysis_week",
        "target_analysis_week",
        "horizon_weeks",
        "model",
    ]


    duplicate_count = int(
        df
        .duplicated(
            subset=duplicate_columns
        )
        .sum()
    )


    # --------------------------------------------------------
    # TEST-YEAR CHECK
    # --------------------------------------------------------

    wrong_target_year = int(
        (
            df[
                "target_year"
            ]
            !=
            df[
                "test_year"
            ]
        )
        .sum()
    )


    # --------------------------------------------------------
    # HORIZON / TARGET-WEEK CHECK
    # --------------------------------------------------------

    expected_target_analysis_week = (
        df[
            "analysis_week"
        ]
        +
        df[
            "horizon_weeks"
        ]
    )


    wrong_target_analysis_week = int(
        (
            df[
                "target_analysis_week"
            ]
            !=
            expected_target_analysis_week
        )
        .sum()
    )


    # --------------------------------------------------------
    # LEGITIMATE CROSS-YEAR FORECASTS
    # --------------------------------------------------------

    cross_year_forecasts = int(
        (
            df[
                "source_year"
            ]
            !=
            df[
                "target_year"
            ]
        )
        .sum()
    )


    # --------------------------------------------------------
    # EXPECTED MODELS
    # --------------------------------------------------------

    observed_models = set(
        models
    )


    expected_models_present = (
        observed_models
        ==
        EXPECTED_MODELS[
            family
        ]
    )


    # --------------------------------------------------------
    # EXPECTED HORIZONS
    # --------------------------------------------------------

    observed_horizons = set(
        horizons
    )


    expected_horizons_present = (
        observed_horizons
        ==
        EXPECTED_HORIZONS
    )


    # --------------------------------------------------------
    # INDIVIDUAL FAMILY PASS
    # --------------------------------------------------------

    family_pass = all(
        [
            missing_actual == 0,
            missing_predicted == 0,
            nonfinite_actual == 0,
            nonfinite_predicted == 0,
            duplicate_count == 0,
            wrong_target_year == 0,
            wrong_target_analysis_week == 0,
            expected_models_present,
            expected_horizons_present,
        ]
    )


    if not family_pass:

        overall_pass = False


    # --------------------------------------------------------
    # PRINT DIAGNOSTICS
    # --------------------------------------------------------

    print(
        "Missing actual:",
        missing_actual,
    )

    print(
        "Missing predicted:",
        missing_predicted,
    )

    print(
        "Non-finite actual:",
        nonfinite_actual,
    )

    print(
        "Non-finite predicted:",
        nonfinite_predicted,
    )

    print(
        "Duplicate forecast rows:",
        duplicate_count,
    )

    print(
        "Rows with target_year != test_year:",
        wrong_target_year,
    )

    print(
        "Rows with incorrect target_analysis_week:",
        wrong_target_analysis_week,
    )

    print(
        "Cross-year forecasts retained:",
        cross_year_forecasts,
    )

    print(
        "Expected models present:",
        expected_models_present,
    )

    print(
        "Expected horizons present:",
        expected_horizons_present,
    )

    print(
        "VALIDATION:",
        (
            "PASS"
            if family_pass
            else "FAIL"
        ),
    )


    # --------------------------------------------------------
    # MODEL/HORIZON COVERAGE
    # --------------------------------------------------------

    print(
        "\nRows by setting / horizon / model:"
    )


    coverage = (
        df
        .groupby(
            [
                "setting",
                "horizon_weeks",
                "model",
            ]
        )
        .size()
        .rename(
            "rows"
        )
        .reset_index()
    )


    print(
        coverage.to_string(
            index=False
        )
    )


    validation_rows.append(
        {
            "family":
                family,

            "rows":
                len(df),

            "settings":
                len(settings),

            "models":
                ", ".join(
                    models
                ),

            "horizons":
                ", ".join(
                    str(x)
                    for x
                    in horizons
                ),

            "missing_actual":
                missing_actual,

            "missing_predicted":
                missing_predicted,

            "nonfinite_actual":
                nonfinite_actual,

            "nonfinite_predicted":
                nonfinite_predicted,

            "duplicates":
                duplicate_count,

            "wrong_target_year":
                wrong_target_year,

            "wrong_target_analysis_week":
                wrong_target_analysis_week,

            "cross_year_forecasts":
                cross_year_forecasts,

            "expected_models_present":
                expected_models_present,

            "expected_horizons_present":
                expected_horizons_present,

            "validation_passed":
                family_pass,
        }
    )


# ============================================================
# SAVE VALIDATION SUMMARY
# ============================================================

validation_df = pd.DataFrame(
    validation_rows
)


validation_file = (
    TABLE_DIR
    / "prediction_split_validation.csv"
)


validation_df.to_csv(
    validation_file,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "OVERALL PREDICTION-SPLIT VALIDATION"
)

print(
    "=" * 70
)


print(
    validation_df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    validation_file,
)


if overall_pass:

    print(
        "\nPREDICTION-SPLIT VALIDATION PASSED"
    )

else:

    raise RuntimeError(
        "Prediction-split validation failed. "
        "Review the diagnostics above before "
        "performing the final model comparison."
    )