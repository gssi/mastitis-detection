"""
Lactose (Lattosio) pipeline.

This script:
- Loads lactose measurements from CSV.
- Restricts to animals present in the functional control (CF) dataset.
- Keeps recent years only (>2018), removes duplicates.
- Fills missing lactose values from 'valoreMisura' when available.
- Applies positivity and IQR-based outlier filtering.
- Aggregates to one measurement per animal-day (first observation).
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- Mirrors filtering choices used in other CF-derived pipelines for consistency.

Outputs:
- ltts_agg.parquet: daily lactose values per animal.
"""

from libraries import pd, Path, log, gc


### PATHS AND STATICS ###


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_LTTS_CSV = PROJECT_ROOT / "db" / "lattosio" / "lattosio.csv"
CF_IDS_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "cf_ids.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "ltts_agg.parquet"

VARIABILE = "Lattosio"

# Column rename map (Italian -> English)
RENAME_MAP = {
    "idAnimale": "id",
    "Lattosio": "lactose",
    "giorno": "day",
    "mese": "month",
    "anno": "year",
}


### HELPERS ###

def IQR_filtering(df: pd.DataFrame, col: str, iqr_k: float = 1.5) -> pd.DataFrame:
    
    """
    Apply Tukey's IQR rule to remove outliers on a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    col : str
        Numeric column name to filter.
    iqr_k : float, default 1.5
        IQR multiplier (1.5 = standard Tukey fences).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (rows outside [Q1 - k*IQR, Q3 + k*IQR] are removed).
    """
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    before = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi)]
    log.info("IQR filter on '%s': %d ➔ %d records", col, before, len(df))
    return df


### MAIN ###

def ltts_main() -> None:
    
    """
    End-to-end processing of lactose measurements.

    Steps:
    1) Load CF IDs universe.
    2) Read lactose CSV; keep years > 2018; drop duplicates.
    3) Filter to CF animals; enforce integer types on date keys.
    4) If present, fill missing 'Lattosio' from 'valoreMisura'.
    5) Keep positive values only; apply IQR filtering.
    6) Aggregate to first measurement per animal-day.
    7) Save final Parquet.
    """
    
    # CF IDs universe
    ids_cf = pd.read_parquet(CF_IDS_PARQUET)["id"].unique()
    log.info("Unique IDs from functional control: %d", len(ids_cf))
    # Load lactose data and base filters
    df = pd.read_csv(
        RAW_LTTS_CSV,
        low_memory=False,
        usecols=lambda c: c not in {"idMisuraPrimaria", "siglaProvincia", "codiceRazzaAIA", "codiceSpecieAIA"},
    )
    df = df.query("anno > 2018").drop_duplicates()
    # Keep only animals present in CF universe
    df = df[df["idAnimale"].isin(ids_cf)]
    log.info("Uploaded data: %s rows, %s columns", f"{df.shape[0]:,}", df.shape[1])
    # Stable temporal sort (mergesort preserves order on ties)
    df = df.sort_values(["idAnimale", "anno", "mese", "giorno"], kind="mergesort")
    # Enforce integer types on keys
    df["idAnimale"] = df["idAnimale"].astype("int64")
    df["giorno"] = df["giorno"].astype("int64")
    df["mese"] = df["mese"].astype("int64")
    df["anno"] = df["anno"].astype("int64")
    # If an alternative value column exists, backfill missing primary values from it
    if "valoreMisura" in df.columns and df[VARIABILE].isna().sum() > 0:
        df[VARIABILE] = df[VARIABILE].fillna(df["valoreMisura"])
    # Keep valid positive lactose values only
    df = df[df[VARIABILE].notna() & (df[VARIABILE] > 0)]
    # Outlier removal via IQR
    df = IQR_filtering(df, VARIABILE)
    # Aggregate to one measurement per animal-day
    df_agg = df.groupby(
        ["idAnimale", "anno", "mese", "giorno"],
        observed=True,
        as_index=False
    ).agg({VARIABILE: "first"})
    # Free memory
    del df
    gc.collect()
    log.info("Aggregation completed – final rows: %s, columns: %s", f"{df_agg.shape[0]:,}", df_agg.shape[1])
    # Save final dataset
    df_agg = df_agg.rename(columns=RENAME_MAP)
    df_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s (%d rows)", OUTPUT_PARQUET, len(df_agg))
    # Cleanup
    del df_agg
    gc.collect()


### ENTRY POINT ###


if __name__ == "__main__":
    ltts_main()



