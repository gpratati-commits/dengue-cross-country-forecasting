"""
Validate the completed weekly dengue calendar.

This script distinguishes inserted calendar rows from
observed surveillance records and summarizes gaps by
setting and calendar year.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/processed/dengue_weekly_clean.csv"
)

OUTPUT_FILE = Path(
    "outputs/tables/inserted_gap_summary_by_year.csv"
)


# ------------------------------------------------------------
# Read processed data
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

print(
    f"Processed rows: {len(df):,}"
)


# ------------------------------------------------------------
# Select inserted rows
# ------------------------------------------------------------

gaps = df[
    df["structural_gap"] == True
].copy()

print(
    f"Inserted rows: {len(gaps):,}"
)


# ------------------------------------------------------------
# Summarize by setting and year
# ------------------------------------------------------------

gap_summary = (
    gaps
    .groupby(
        [
            "setting",
            "iso_year"
        ]
    )
    .agg(
        inserted_weeks=(
            "iso_week",
            "count"
        ),
        first_inserted_week=(
            "iso_week",
            "min"
        ),
        last_inserted_week=(
            "iso_week",
            "max"
        )
    )
    .reset_index()
)


print(
    "\nINSERTED GAP SUMMARY BY YEAR"
)

print(
    gap_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Show all week-53 insertions separately
# ------------------------------------------------------------

week53 = gaps[
    gaps["iso_week"] == 53
][
    [
        "setting",
        "date",
        "iso_year",
        "iso_week"
    ]
]

print(
    "\nINSERTED ISO WEEK-53 ROWS"
)

if week53.empty:

    print("None")

else:

    print(
        week53.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Check missing case values
# ------------------------------------------------------------

nonmissing_cases = gaps[
    gaps["cases"].notna()
]

if not nonmissing_cases.empty:

    raise AssertionError(
        "Inserted rows unexpectedly contain dengue cases."
    )


print(
    "\nAll inserted rows have missing dengue cases: PASS"
)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

gap_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved: {OUTPUT_FILE}"
)