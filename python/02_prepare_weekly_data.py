"""
Prepare weekly dengue surveillance and climate data for analysis.

Important:
The source workbook documents a weekly-alignment ambiguity:
environmental series are described as continuous 7-day blocks, with
rare 53rd blocks folded into week 52, although WN is also described
elsewhere as an ISO week number.

For this reason, this script does NOT create missing ISO week-53 rows.

Instead, it uses the source YR/WN fields directly and assumes an
analysis calendar of 52 source-week blocks per year.

Missing surveillance rows remain NaN and are never converted to zero.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

RAW_FILE = Path(
    "data/raw/Dengue_Environmental Vars_Dataset.xlsx"
)

SHEET_NAME = "Multi-Setting Data"

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/tables")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. READ RAW DATA
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}"
    )

df = pd.read_excel(
    RAW_FILE,
    sheet_name=SHEET_NAME
)

print("Raw dataset loaded successfully.")
print(f"Original rows: {len(df):,}")


# ============================================================
# 3. COLUMNS
# ============================================================

required_columns = [
    "SETTING",
    "YR",
    "WN",
    "DC_OPENDENGUE",
    "RF_NASA",
    "RF_ERA5-Land",
    "Temp_ERA5-Land",
    "SH_ERA5-Land",
    "Temp_MERRA2",
    "TempMax_MERRA2",
    "TempMin_MERRA2",
    "SH_MERRA2",
]

optional_columns = [
    "FLAG_SINGLE_CELL_RF",
    "FLAG_TERMINAL_GAP",
    "FLAG_MERRA2_SH_LOW_COVERAGE",
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_required)
    )

columns_to_keep = required_columns + [
    col
    for col in optional_columns
    if col in df.columns
]

data = df[columns_to_keep].copy()


# ============================================================
# 4. RENAME VARIABLES
# ============================================================

data = data.rename(
    columns={
        "SETTING": "setting",
        "YR": "source_year",
        "WN": "source_week",
        "DC_OPENDENGUE": "cases",
        "RF_NASA": "rain_nasa",
        "RF_ERA5-Land": "rain_era5",
        "Temp_ERA5-Land": "temp_era5",
        "SH_ERA5-Land": "humidity_era5",
        "Temp_MERRA2": "temp_merra2",
        "TempMax_MERRA2": "tempmax_merra2",
        "TempMin_MERRA2": "tempmin_merra2",
        "SH_MERRA2": "humidity_merra2",
        "FLAG_SINGLE_CELL_RF": "flag_single_cell_rf",
        "FLAG_TERMINAL_GAP": "flag_terminal_gap",
        "FLAG_MERRA2_SH_LOW_COVERAGE":
            "flag_merra2_humidity_low_coverage",
    }
)


# ============================================================
# 5. STANDARDIZE TYPES
# ============================================================

data["setting"] = (
    data["setting"]
    .astype(str)
    .str.strip()
    .str.title()
)

data["source_year"] = pd.to_numeric(
    data["source_year"],
    errors="raise"
).astype(int)

data["source_week"] = pd.to_numeric(
    data["source_week"],
    errors="raise"
).astype(int)


# ============================================================
# 6. CHECK SOURCE WEEK RANGE
# ============================================================

invalid_week = ~data["source_week"].between(
    1,
    52
)

if invalid_week.any():

    print(
        data.loc[
            invalid_week,
            [
                "setting",
                "source_year",
                "source_week"
            ]
        ]
    )

    raise ValueError(
        "Source weeks outside the expected 1-52 range."
    )

print(
    "All source weeks fall within 1-52."
)


# ============================================================
# 7. CHECK DUPLICATES
# ============================================================

duplicate_mask = data.duplicated(
    subset=[
        "setting",
        "source_year",
        "source_week"
    ],
    keep=False
)

if duplicate_mask.any():

    print(
        data.loc[
            duplicate_mask,
            [
                "setting",
                "source_year",
                "source_week"
            ]
        ]
    )

    raise ValueError(
        "Duplicate setting-year-week rows found."
    )

print(
    "No duplicate setting-year-week rows detected."
)


# ============================================================
# 8. MARK ORIGINAL SOURCE ROWS
# ============================================================

data["source_row_present"] = True


# ============================================================
# 9. COMPLETE SOURCE-WEEK GRID
# ============================================================

completed_parts = []

for setting_name, group in data.groupby(
    "setting",
    sort=False
):

    min_year = int(
        group["source_year"].min()
    )

    max_year = int(
        group["source_year"].max()
    )

    expected_grid = pd.MultiIndex.from_product(
        [
            range(
                min_year,
                max_year + 1
            ),
            range(1, 53)
        ],
        names=[
            "source_year",
            "source_week"
        ]
    )

    completed_group = (
        group
        .set_index(
            [
                "source_year",
                "source_week"
            ]
        )
        .reindex(expected_grid)
        .reset_index()
    )

    completed_group["setting"] = setting_name

    completed_group["source_row_present"] = (
        completed_group[
            "source_row_present"
        ]
        .eq(True)
    )

    completed_parts.append(
        completed_group
    )


completed = pd.concat(
    completed_parts,
    ignore_index=True
)


# ============================================================
# 10. STRUCTURAL-GAP INDICATORS
# ============================================================

completed["structural_gap"] = (
    ~completed["source_row_present"]
)

completed["dengue_missing"] = (
    completed["cases"].isna()
)


# ============================================================
# 11. CREATE A CONTINUOUS ANALYSIS-WEEK INDEX
# ============================================================

global_start_year = int(
    completed["source_year"].min()
)

completed["analysis_week"] = (
    (
        completed["source_year"]
        - global_start_year
    )
    * 52
    + completed["source_week"]
)


# Human-readable label

completed["week_label"] = (
    completed["source_year"]
    .astype(str)
    + "-W"
    + completed["source_week"]
    .astype(str)
    .str
    .zfill(2)
)


# ============================================================
# 12. SORT
# ============================================================

completed = (
    completed
    .sort_values(
        [
            "setting",
            "analysis_week"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. SUMMARY
# ============================================================

summary = (
    completed
    .groupby("setting")
    .agg(
        total_weekly_rows=(
            "analysis_week",
            "size"
        ),
        original_rows=(
            "source_row_present",
            "sum"
        ),
        inserted_gap_rows=(
            "structural_gap",
            "sum"
        ),
        missing_dengue_rows=(
            "dengue_missing",
            "sum"
        )
    )
    .reset_index()
)

print(
    "\nWeekly source-calendar summary:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 14. GAP SUMMARY BY YEAR
# ============================================================

gap_summary = (
    completed[
        completed["structural_gap"]
    ]
    .groupby(
        [
            "setting",
            "source_year"
        ]
    )
    .agg(
        inserted_weeks=(
            "source_week",
            "count"
        ),
        first_inserted_week=(
            "source_week",
            "min"
        ),
        last_inserted_week=(
            "source_week",
            "max"
        )
    )
    .reset_index()
)

print(
    "\nSTRUCTURAL GAP SUMMARY"
)

print(
    gap_summary.to_string(
        index=False
    )
)


# ============================================================
# 15. SAFETY CHECK
# ============================================================

gap_rows = completed[
    completed["structural_gap"]
]

if gap_rows["cases"].notna().any():

    raise AssertionError(
        "Inserted structural gaps unexpectedly "
        "contain dengue cases."
    )

print(
    "\nSafety check passed:"
)

print(
    "Inserted surveillance gaps remain NaN, "
    "not zero."
)


# ============================================================
# 16. SAVE
# ============================================================

processed_file = (
    PROCESSED_DIR
    / "dengue_weekly_clean.csv"
)

completed.to_csv(
    processed_file,
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "weekly_calendar_completion_summary.csv",
    index=False
)

gap_summary.to_csv(
    OUTPUT_DIR
    / "structural_gap_summary_by_year.csv",
    index=False
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "WEEKLY DATA PREPARATION COMPLETE"
)

print(
    "=" * 60
)

print(
    f"Original rows: {len(data):,}"
)

print(
    f"Completed source-week rows: "
    f"{len(completed):,}"
)

print(
    f"Inserted structural-gap rows: "
    f"{int(completed['structural_gap'].sum()):,}"
)

print(
    "\nProcessed dataset:"
)

print(
    processed_file
)