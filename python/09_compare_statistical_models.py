"""
Compare simple baselines and negative-binomial forecasting models.

Creates:
1. Model-performance comparison by setting/horizon.
2. Improvement relative to persistence.
3. Climate-ablation comparison within the negative-binomial models.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("outputs/tables")

BASELINE_FILE = (
    TABLE_DIR
    / "baseline_forecasting_results.csv"
)

NB_FILE = (
    TABLE_DIR
    / "negative_binomial_results.csv"
)


# ============================================================
# READ RESULTS
# ============================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

nb = pd.read_csv(
    NB_FILE
)


# ============================================================
# EXTRACT PERSISTENCE BENCHMARK
# ============================================================

persistence = (
    baseline[
        baseline["model"] == "persistence"
    ]
    [
        [
            "setting",
            "horizon_weeks",
            "mae",
            "rmse",
            "mase",
        ]
    ]
    .rename(
        columns={
            "mae": "persistence_mae",
            "rmse": "persistence_rmse",
            "mase": "persistence_mase",
        }
    )
)


# ============================================================
# ADD PERSISTENCE TO NB RESULTS
# ============================================================

comparison = nb.merge(
    persistence,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="left",
)


# ============================================================
# IMPROVEMENT RELATIVE TO PERSISTENCE
# ============================================================

comparison[
    "mase_improvement_vs_persistence"
] = (
    comparison["persistence_mase"]
    - comparison["mase"]
)


comparison[
    "mase_percent_improvement_vs_persistence"
] = (
    (
        comparison["persistence_mase"]
        - comparison["mase"]
    )
    / comparison["persistence_mase"]
    * 100
)


comparison[
    "beats_persistence"
] = (
    comparison["mase"]
    < comparison["persistence_mase"]
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_file = (
    TABLE_DIR
    / "statistical_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# CLIMATE ABLATION
# ============================================================

epi = (
    nb[
        nb["model"]
        == "epidemiology_only"
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
                "mase_epidemiology_only"
        }
    )
)


climate = (
    nb[
        nb["model"]
        == "climate_informed"
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
                "mase_climate_informed"
        }
    )
)


ablation = epi.merge(
    climate,
    on=[
        "setting",
        "horizon_weeks",
    ],
    how="inner",
)


ablation[
    "delta_mase_climate_minus_epi"
] = (
    ablation[
        "mase_climate_informed"
    ]
    - ablation[
        "mase_epidemiology_only"
    ]
)


ablation[
    "climate_improves_forecast"
] = (
    ablation[
        "delta_mase_climate_minus_epi"
    ]
    < 0
)


ablation_file = (
    TABLE_DIR
    / "negative_binomial_climate_ablation.csv"
)

ablation.to_csv(
    ablation_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    comparison
    .groupby(
        [
            "horizon_weeks",
            "model",
        ],
        as_index=False,
    )
    .agg(
        median_mase=(
            "mase",
            "median"
        ),
        median_percent_improvement=(
            "mase_percent_improvement_vs_persistence",
            "median"
        ),
        settings_beating_persistence=(
            "beats_persistence",
            "sum"
        ),
        settings_evaluated=(
            "setting",
            "nunique"
        ),
    )
)


summary_file = (
    TABLE_DIR
    / "statistical_model_comparison_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


print(
    "\nSTATISTICAL MODEL COMPARISON"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nNEGATIVE-BINOMIAL CLIMATE ABLATION"
)

print(
    ablation.to_string(
        index=False
    )
)


print(
    "\nMODEL COMPARISON COMPLETE"
)

print(
    f"Saved: {comparison_file}"
)

print(
    f"Saved: {ablation_file}"
)

print(
    f"Saved: {summary_file}"
)