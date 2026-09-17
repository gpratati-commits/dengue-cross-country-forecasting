"""
Exploratory climate-lag screening for dengue.

For each setting, climate variable, and lag, fit:

log(1 + cases_t) ~
    log(1 + cases_{t-1})
    + seasonal sine/cosine terms
    + standardized climate exposure

HAC/Newey-West standard errors are used.

P-values are adjusted across all screening tests using the
Benjamini-Hochberg false-discovery-rate procedure.

This is exploratory association screening, not causal inference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/dengue_model_features.csv"
)

TABLE_DIR = Path(
    "outputs/tables"
)

FIGURE_DIR = Path(
    "outputs/figures"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
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

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# CLIMATE VARIABLES AND LAGS
# ============================================================

CLIMATE_VARIABLES = {
    "rainfall": "log_rain_era5",
    "temperature": "temp_era5",
    "humidity": "humidity_era5",
}

LAGS = [
    0,
    1,
    2,
    4,
    6,
    8,
]


# ============================================================
# FIT ONE SCREENING MODEL
# ============================================================

def fit_lag_model(
    group,
    climate_column
):

    model_df = group[
        [
            "log_cases",
            "log_cases_lag_1",
            "season_sin",
            "season_cos",
            climate_column,
        ]
    ].dropna().copy()

    if len(model_df) < 30:
        return None

    climate_sd = (
        model_df[
            climate_column
        ].std()
    )

    if (
        pd.isna(climate_sd)
        or climate_sd == 0
    ):
        return None

    model_df["climate_z"] = (
        (
            model_df[
                climate_column
            ]
            - model_df[
                climate_column
            ].mean()
        )
        / climate_sd
    )

    X = model_df[
        [
            "log_cases_lag_1",
            "season_sin",
            "season_cos",
            "climate_z",
        ]
    ]

    X = sm.add_constant(X)

    y = model_df[
        "log_cases"
    ]

    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": 4
        }
    )

    return {
        "n": len(model_df),
        "beta": model.params[
            "climate_z"
        ],
        "se": model.bse[
            "climate_z"
        ],
        "p_value": model.pvalues[
            "climate_z"
        ],
        "adjusted_r2":
            model.rsquared_adj,
    }


# ============================================================
# RUN ALL SCREENING MODELS
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

    for climate_name, base_name in (
        CLIMATE_VARIABLES.items()
    ):

        for lag in LAGS:

            climate_column = (
                f"{base_name}_lag_{lag}"
            )

            if (
                climate_column
                not in group.columns
            ):
                continue

            result = fit_lag_model(
                group,
                climate_column
            )

            if result is None:
                continue

            results.append(
                {
                    "setting":
                        setting,

                    "climate_variable":
                        climate_name,

                    "lag_weeks":
                        lag,

                    **result,
                }
            )


results_df = pd.DataFrame(
    results
)

if results_df.empty:
    raise RuntimeError(
        "No climate-lag models were fitted."
    )


# ============================================================
# MULTIPLE-TESTING CORRECTION
# ============================================================

reject_fdr, q_values, _, _ = (
    multipletests(
        results_df[
            "p_value"
        ],
        alpha=0.05,
        method="fdr_bh",
    )
)

results_df[
    "q_value_fdr_bh"
] = q_values

results_df[
    "significant_fdr_0_05"
] = reject_fdr


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

results_df[
    "ci_lower"
] = (
    results_df["beta"]
    - 1.96
    * results_df["se"]
)

results_df[
    "ci_upper"
] = (
    results_df["beta"]
    + 1.96
    * results_df["se"]
)


# ============================================================
# SAVE FULL RESULTS
# ============================================================

screening_file = (
    TABLE_DIR
    / "climate_lag_screening.csv"
)

results_df.to_csv(
    screening_file,
    index=False
)

print(
    f"\nModels fitted: "
    f"{len(results_df):,}"
)

print(
    f"Saved: {screening_file}"
)


# ============================================================
# BEST LAG PER SETTING / CLIMATE VARIABLE
# ============================================================

best_lags = (
    results_df
    .sort_values(
        "p_value"
    )
    .groupby(
        [
            "setting",
            "climate_variable",
        ],
        as_index=False,
    )
    .first()
)

best_file = (
    TABLE_DIR
    / "climate_lag_best_lags.csv"
)

best_lags.to_csv(
    best_file,
    index=False
)

print(
    "\nLowest-p-value lag for each "
    "setting/climate variable:"
)

print(
    best_lags[
        [
            "setting",
            "climate_variable",
            "lag_weeks",
            "beta",
            "p_value",
            "q_value_fdr_bh",
            "significant_fdr_0_05",
            "n",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY OF FDR-SIGNIFICANT RESULTS
# ============================================================

fdr_significant = (
    results_df[
        results_df[
            "significant_fdr_0_05"
        ]
    ]
    .sort_values(
        [
            "setting",
            "climate_variable",
            "lag_weeks",
        ]
    )
)

print(
    "\nFDR-significant climate-lag "
    "associations (q < 0.05):"
)

if fdr_significant.empty:

    print(
        "None."
    )

else:

    print(
        fdr_significant[
            [
                "setting",
                "climate_variable",
                "lag_weeks",
                "beta",
                "p_value",
                "q_value_fdr_bh",
                "n",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FIGURES
# ============================================================

for climate_name in (
    CLIMATE_VARIABLES.keys()
):

    plot_data = (
        results_df[
            results_df[
                "climate_variable"
            ]
            == climate_name
        ]
        .copy()
    )

    plt.figure(
        figsize=(10, 6)
    )

    for setting, group in (
        plot_data.groupby(
            "setting"
        )
    ):

        group = (
            group
            .sort_values(
                "lag_weeks"
            )
        )

        plt.plot(
            group[
                "lag_weeks"
            ],
            group[
                "beta"
            ],
            marker="o",
            label=setting,
        )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Climate lag (weeks)"
    )

    plt.ylabel(
        "Standardized climate coefficient"
    )

    plt.title(
        "Adjusted dengue association "
        f"with {climate_name}"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    figure_file = (
        FIGURE_DIR
        / (
            f"climate_lag_"
            f"{climate_name}.png"
        )
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nCLIMATE-LAG SCREENING COMPLETE"
)