from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "data/processed/dengue_model_features.csv"
)

OUTPUT_FILE = Path(
    "outputs/tables/temporal_leakage_audit.csv"
)

HORIZONS = [
    1,
    2,
    4,
]

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


required = [
    "setting",
    "source_year",
    "source_week",
    "analysis_week",
    "cases",
]


missing = [
    column
    for column in required
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


df = (
    df
    .sort_values(
        [
            "setting",
            "analysis_week",
        ]
    )
    .copy()
)


# ============================================================
# CREATE TARGET-TIME LOOKUP
# ============================================================

target_lookup = (
    df[
        [
            "setting",
            "analysis_week",
            "source_year",
            "source_week",
            "cases",
        ]
    ]
    .rename(
        columns={
            "analysis_week":
                "target_analysis_week",

            "source_year":
                "target_year",

            "source_week":
                "target_week",

            "cases":
                "target_cases",
        }
    )
)


audit_rows = []


# ============================================================
# AUDIT EACH SETTING
# ============================================================

for setting in sorted(
    df["setting"]
    .dropna()
    .unique()
):

    group = (
        df[
            df["setting"] == setting
        ]
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


    if len(observed_years) < 3:

        continue


    validation_year = (
        observed_years[-2]
    )

    test_year = (
        observed_years[-1]
    )


    for horizon in HORIZONS:

        origins = (
            group[
                [
                    "setting",
                    "analysis_week",
                    "source_year",
                    "source_week",
                ]
            ]
            .copy()
        )


        origins[
            "target_analysis_week"
        ] = (
            origins[
                "analysis_week"
            ]
            + horizon
        )


        check = origins.merge(
            target_lookup,
            on=[
                "setting",
                "target_analysis_week",
            ],
            how="left"
        )


        check = check[
            check["target_cases"].notna()
        ].copy()


        # ----------------------------------------
        # General cross-year forecast
        # ----------------------------------------

        check[
            "crosses_calendar_year"
        ] = (
            check["source_year"]
            != check["target_year"]
        )


        # ----------------------------------------
        # Potential training -> validation leakage
        # ----------------------------------------

        check[
            "train_to_validation_leak"
        ] = (
            (
                check["source_year"]
                < validation_year
            )
            &
            (
                check["target_year"]
                >= validation_year
            )
        )


        # ----------------------------------------
        # Potential validation -> test leakage
        # ----------------------------------------

        check[
            "validation_to_test_leak"
        ] = (
            (
                check["source_year"]
                == validation_year
            )
            &
            (
                check["target_year"]
                >= test_year
            )
        )


        audit_rows.append(
            {
                "setting":
                    setting,

                "horizon_weeks":
                    horizon,

                "validation_year":
                    validation_year,

                "test_year":
                    test_year,

                "forecast_pairs":
                    len(check),

                "cross_year_pairs":
                    int(
                        check[
                            "crosses_calendar_year"
                        ].sum()
                    ),

                "train_to_validation_leak":
                    int(
                        check[
                            "train_to_validation_leak"
                        ].sum()
                    ),

                "validation_to_test_leak":
                    int(
                        check[
                            "validation_to_test_leak"
                        ].sum()
                    ),
            }
        )


# ============================================================
# RESULTS
# ============================================================

audit = pd.DataFrame(
    audit_rows
)


if audit.empty:

    raise RuntimeError(
        "Temporal audit produced no rows."
    )


audit.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nTEMPORAL LEAKAGE AUDIT"
)


print(
    audit.to_string(
        index=False
    )
)


print(
    "\nTOTALS"
)


print(
    "Cross-year forecast pairs:",
    audit[
        "cross_year_pairs"
    ].sum()
)


print(
    "Training -> validation "
    "boundary violations:",
    audit[
        "train_to_validation_leak"
    ].sum()
)


print(
    "Validation -> test "
    "boundary violations:",
    audit[
        "validation_to_test_leak"
    ].sum()
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print(
    "\nTEMPORAL LEAKAGE AUDIT COMPLETE"
)