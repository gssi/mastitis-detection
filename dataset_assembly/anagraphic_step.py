"""
Data pipeline for animal registry (Anagrafica).

This script:
- Loads animal registry records (anagraphic) from raw CSV.
- Keeps only animals present in the functional check dataset (cf_ids).
- Filters birth dates for validity and reasonable years.
- Resolves duplicates and enforces one unique birth date per animal.
- Saves a clean table with animal IDs and their birth dates.

Context:
- Part of the mammary diseases indicators workflow.
- Ensures consistency between functional check records and registry information.

Outputs:
- ana_agg.parquet: mapping of animal IDs to validated birth dates
"""

from libraries import pd, Path, log, gc

# =========================
# PATHS AND CONSTANTS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ANA_CSV = PROJECT_ROOT / "db" / "anagrafica" / "ana.csv"
CF_IDS_PARQUET = (
    PROJECT_ROOT
    / "mammary_diseases_indicators"
    / "temporary_datasets"
    / "cf_ids.parquet"
)
OUTPUT_PARQUET = (
    PROJECT_ROOT
    / "mammary_diseases_indicators"
    / "temporary_datasets"
    / "ana_agg.parquet"
)

# Column rename map (Italian -> English)
RENAME_MAP = {
    "idAnimale": "id",
    "DataNascita": "birth_date",
}

# =========================
# MAIN FUNCTION
# =========================

def ana_main():
    
    """
    Extract, filter, and validate animal registry (anagraphic) data.

    Steps:
    1) Load unique IDs from the functional check dataset (cf_ids).
    2) Load registry CSV, keeping only those IDs.
    3) Parse and filter birth dates to plausible years (2018–2022).
    4) Resolve duplicates, keeping a unique birth date per animal.
    5) Save the final validated dataset.

    Output:
    - A Parquet file with two columns: ['id', 'birth_date'].
    """
    
    # Load IDs from functional check
    log.info("Starting elaboration...")
    ids_cf = pd.read_parquet(CF_IDS_PARQUET)["id"].unique()
    log.info("Unique IDs from functional check: %d", len(ids_cf))
    # Load anagraphic data
    log.info("Loading anagraphic data from %s", RAW_ANA_CSV)
    df = pd.read_csv(
        RAW_ANA_CSV,
        low_memory=False,
        # Drop unnecessary columns at load time to save memory
        usecols=lambda c: c
        not in {"idMisuraPrimaria", "siglaProvincia", "codiceRazzaAIA", "codiceSpecieAIA"},
    )
    # Keep only animals present in the functional check
    df = df[df["idAnimale"].isin(ids_cf)]
    log.info("Data extracted: %s rows, %s columns", f"{df.shape[0]:,}", df.shape[1])
    # Birth date cleaning
    df["DataNascita"] = pd.to_datetime(df["DataNascita"], errors="coerce")
    df = df[df["DataNascita"].notna()]
    # Keep plausible years only (2018–2022)
    df = df[(df["DataNascita"].dt.year > 2017) & (df["DataNascita"].dt.year < 2023)]
    # Remove duplicates and enforce consistent type
    df = df.drop_duplicates()
    df["idAnimale"] = df["idAnimale"].astype("int64")
    # Aggregate per animal: earliest birth date (safety check)
    nascita_agg = df.groupby("idAnimale", as_index=False)["DataNascita"].min()
    # Re-merge to ensure one row per unique animal
    nascita_agg = pd.merge(
        nascita_agg,
        df[["idAnimale"]].drop_duplicates("idAnimale"),
        on="idAnimale",
        how="left",
    )
    # Check for consistency: keep only animals with exactly one valid birth date
    conteggio = nascita_agg["idAnimale"].value_counts()
    id_unici = conteggio[conteggio == 1].index
    nascita_agg = nascita_agg[nascita_agg["idAnimale"].isin(id_unici)]
    log.info("Valid animals: %d", len(nascita_agg))
    # Save final dataset
    nascita_agg = nascita_agg.rename(columns=RENAME_MAP)
    nascita_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s (%d rows)", OUTPUT_PARQUET, len(nascita_agg))
    # Cleanup
    del df, nascita_agg, conteggio, id_unici
    gc.collect()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    ana_main()


