from pathlib import Path
import pandas as pd

# -----------------------------
# File paths
# -----------------------------

RAW_FILE = Path(
    "data/raw/Dengue_Environmental Vars_Dataset.xlsx"
)

SHEET_NAME = "Multi-Setting Data"


# -----------------------------
# Check file
# -----------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_FILE}"
    )

print("Dataset found.")


# -----------------------------
# Read dataset
# -----------------------------

df = pd.read_excel(
    RAW_FILE,
    sheet_name=SHEET_NAME
)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())
# ============================================================
# RESEARCH-GRADE DATA QUALITY CONTROL
# ============================================================

print("\n" + "=" * 60)
print("DENGUE DATA QUALITY CONTROL SUMMARY")
print("=" * 60)


# ------------------------------------------------------------
# 1. Number of records
# ------------------------------------------------------------

n_records = len(df)

print(f"\n1. Number of records: {n_records:,}")


# ------------------------------------------------------------
# 2. Number of countries / settings
# ------------------------------------------------------------

settings = sorted(
    df["SETTING"]
    .dropna()
    .astype(str)
    .unique()
)

print(f"\n2. Number of settings: {len(settings)}")

print("Settings:")

for setting in settings:
    print(f"   - {setting}")


# ------------------------------------------------------------
# 3. Duplicate setting-year-week records
# ------------------------------------------------------------

key_columns = [
    "SETTING",
    "YR",
    "WN"
]

duplicate_mask = df.duplicated(
    subset=key_columns,
    keep=False
)

n_duplicates = int(
    duplicate_mask.sum()
)

print(
    "\n3. Duplicate setting-year-week records:",
    n_duplicates
)

if n_duplicates > 0:

    print("\nDuplicated records:")

    print(
        df.loc[
            duplicate_mask,
            key_columns
        ]
        .sort_values(key_columns)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 4. Negative dengue case counts
# ------------------------------------------------------------

case_column = "DC_OPENDENGUE"

cases_numeric = pd.to_numeric(
    df[case_column],
    errors="coerce"
)

negative_case_mask = (
    cases_numeric < 0
)

n_negative_cases = int(
    negative_case_mask.sum()
)

print(
    "\n4. Negative dengue case counts:",
    n_negative_cases
)


# ------------------------------------------------------------
# 5. Missing dengue observations
# ------------------------------------------------------------

missing_dengue = int(
    df[case_column]
    .isna()
    .sum()
)

print(
    "\n5. Missing dengue observations:",
    missing_dengue
)


# ------------------------------------------------------------
# 6. Missing climate observations
# ------------------------------------------------------------

candidate_climate_columns = [
    "RF_NASA",
    "RF_ERA5-Land",
    "Temp_ERA5-Land",
    "SH_ERA5-Land",
    "Temp_MERRA2",
    "TempMax_MERRA2",
    "TempMin_MERRA2",
    "SH_MERRA2"
]

climate_columns = [
    col
    for col in candidate_climate_columns
    if col in df.columns
]

print("\n6. Missing climate observations:")

climate_missing = (
    df[climate_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(climate_missing)


# ------------------------------------------------------------
# 7. Year coverage
# ------------------------------------------------------------

year_coverage = (
    df
    .groupby("SETTING")
    .agg(
        first_year=("YR", "min"),
        last_year=("YR", "max"),
        observed_years=("YR", "nunique")
    )
    .sort_index()
)

print("\n7. Year coverage by setting:")

print(year_coverage)


# ------------------------------------------------------------
# 8. Missing years within each setting
# ------------------------------------------------------------

print("\n8. Missing years within each setting:")

missing_year_records = []

for setting, group in df.groupby("SETTING"):

    observed_years = sorted(
        group["YR"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(observed_years) == 0:
        continue

    expected_years = set(
        range(
            min(observed_years),
            max(observed_years) + 1
        )
    )

    missing_years = sorted(
        expected_years -
        set(observed_years)
    )

    missing_year_records.append(
        {
            "SETTING": setting,
            "missing_years":
                ", ".join(
                    map(str, missing_years)
                )
                if missing_years
                else "None"
        }
    )

    print(
        f"   {setting}: "
        f"{missing_years if missing_years else 'None'}"
    )

missing_years_df = pd.DataFrame(
    missing_year_records
)


# ------------------------------------------------------------
# 9. Week coverage by setting and year
# ------------------------------------------------------------

week_coverage = (
    df
    .groupby(
        ["SETTING", "YR"]
    )
    .agg(
        first_week=("WN", "min"),
        last_week=("WN", "max"),
        observed_weeks=("WN", "nunique")
    )
    .reset_index()
)

print("\n9. Week coverage:")

print(
    week_coverage.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 10. Internal structural gaps in week numbers
# ------------------------------------------------------------

print("\n10. Internal structural gaps:")

structural_gaps = []

for (setting, year), group in df.groupby(
    ["SETTING", "YR"]
):

    weeks = sorted(
        group["WN"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(weeks) == 0:
        continue

    expected_internal_weeks = set(
        range(
            min(weeks),
            max(weeks) + 1
        )
    )

    missing_weeks = sorted(
        expected_internal_weeks -
        set(weeks)
    )

    if missing_weeks:

        structural_gaps.append(
            {
                "SETTING": setting,
                "YR": int(year),
                "missing_weeks":
                    ", ".join(
                        map(str, missing_weeks)
                    )
            }
        )

        print(
            f"   {setting}, {int(year)}: "
            f"{missing_weeks}"
        )

if not structural_gaps:

    print(
        "   No internal missing week numbers detected."
    )


# ------------------------------------------------------------
# 11. Save QC tables
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "outputs/tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

year_coverage.to_csv(
    OUTPUT_DIR / "year_coverage.csv"
)

week_coverage.to_csv(
    OUTPUT_DIR / "week_coverage.csv",
    index=False
)

missing_years_df.to_csv(
    OUTPUT_DIR / "missing_years.csv",
    index=False
)

climate_missing.rename(
    "missing_observations"
).to_csv(
    OUTPUT_DIR /
    "climate_missingness.csv"
)

if structural_gaps:

    pd.DataFrame(
        structural_gaps
    ).to_csv(
        OUTPUT_DIR /
        "structural_week_gaps.csv",
        index=False
    )


# ------------------------------------------------------------
# 12. Final QC message
# ------------------------------------------------------------

print("\n" + "=" * 60)

print("QC COMPLETE")

print(
    "Tables saved in outputs/tables/"
)

print("=" * 60)