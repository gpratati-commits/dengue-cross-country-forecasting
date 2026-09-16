"""
Exploratory epidemiological analysis of the cleaned
multi-setting dengue dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/dengue_weekly_clean.csv"
)

FIGURE_DIR = Path(
    "outputs/figures"
)

TABLE_DIR = Path(
    "outputs/tables"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# BASIC EPIDEMIOLOGICAL SUMMARY
# ============================================================

summary = (
    df
    .groupby("setting")
    .agg(
        total_calendar_weeks=(
            "analysis_week",
            "size"
        ),
        observed_case_weeks=(
            "cases",
            "count"
        ),
        missing_case_weeks=(
            "cases",
            lambda x: x.isna().sum()
        ),
        total_reported_cases=(
            "cases",
            "sum"
        ),
        mean_weekly_cases=(
            "cases",
            "mean"
        ),
        median_weekly_cases=(
            "cases",
            "median"
        ),
        maximum_weekly_cases=(
            "cases",
            "max"
        )
    )
    .reset_index()
)

summary["missing_percent"] = (
    summary["missing_case_weeks"]
    /
    summary["total_calendar_weeks"]
    * 100
)

print(
    "\nEPIDEMIOLOGICAL SUMMARY"
)

print(
    summary.to_string(
        index=False
    )
)

summary.to_csv(
    TABLE_DIR /
    "epidemiological_summary_by_setting.csv",
    index=False
)
# ============================================================
# YEARLY DENGUE BURDEN
# ============================================================

yearly_cases = (
    df
    .groupby(
        [
            "setting",
            "source_year"
        ],
        as_index=False
    )
    .agg(
        reported_cases=(
            "cases",
            "sum"
        ),
        observed_weeks=(
            "cases",
            "count"
        )
    )
)

# Do not interpret years with no observed surveillance
# as zero-case years.

yearly_cases.loc[
    yearly_cases["observed_weeks"] == 0,
    "reported_cases"
] = np.nan

yearly_cases.to_csv(
    TABLE_DIR /
    "yearly_dengue_cases.csv",
    index=False
)

print(
    "\nYEARLY CASE SUMMARY"
)

print(
    yearly_cases.to_string(
        index=False
    )
)
# ============================================================
# SEASONAL PROFILE
# ============================================================

seasonal_profile = (
    df
    .dropna(
        subset=["cases"]
    )
    .groupby(
        [
            "setting",
            "source_week"
        ],
        as_index=False
    )
    .agg(
        median_cases=(
            "cases",
            "median"
        ),
        mean_cases=(
            "cases",
            "mean"
        )
    )
)

seasonal_profile.to_csv(
    TABLE_DIR /
    "seasonal_dengue_profile.csv",
    index=False
)


for setting, group in seasonal_profile.groupby(
    "setting"
):

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        group["source_week"],
        group["median_cases"]
    )

    plt.xlabel(
        "Source week"
    )

    plt.ylabel(
        "Median reported dengue cases"
    )

    plt.title(
        f"Seasonal dengue pattern — {setting}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR /
        (
            "seasonality_"
            + setting.lower().replace(
                " ",
                "_"
            )
            + ".png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()