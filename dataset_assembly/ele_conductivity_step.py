"""
Milk Electrical Conductivity (EC) pipeline.

This script:
- Loads raw EC measurements from CSV.
- Restricts to animals present in the functional check (CF) universe.
- Cleans/normalizes date and ID fields, fills EC from fallback column when needed.
- Removes non-positive/NaN values and applies IQR-based outlier filtering.
- Aggregates to one EC value per animal-day (stable 'first' after sorting).
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- EC is often used as a proxy for subclinical mastitis risk; here we prepare
  a consistent daily signal for downstream modeling.

Outputs:
- ce_agg.parquet: daily EC per animal after filtering.
"""

from libraries import pd, Path, log, gc

# =========================
# PATHS AND STATICS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CE_CSV = PROJECT_ROOT / "db" / "conducibilita_elettrica_latte" / "conducibilita_latte.csv"
CF_IDS_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "cf_ids.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "ce_agg.parquet"

# Name of the EC column in the raw file (accented header preserved)
VARIABILE = "Conducibilità elettrica"

# Column rename map (Italian -> English)
RENAME_MAP = {
    "idAnimale": "id",
    "Conducibilità elettrica": "ec",
    "giorno": "day",
    "mese": "month",
    "anno": "year",
}


# =========================
# HELPERS
# =========================

def IQR_filtering(df: pd.DataFrame, col: str, iqr_k: float = 1.5) -> pd.DataFrame:
    """
    Apply Tukey's IQR rule to remove outliers on a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    col : str
        Column name on which to apply filtering.
    iqr_k : float, default 1.5
        Multiplier for IQR (1.5 = standard Tukey fences; increase to be less strict).

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


# =========================
# MAIN
# =========================

def ec_main() -> None:
    """
    End-to-end processing of milk electrical conductivity (EC).

    Steps:
    1) Load CF IDs universe.
    2) Load EC CSV; keep years > 2018 and deduplicate.
    3) Filter to CF animals; enforce integer dtypes and temporal sorting.
    4) Fill EC from 'valoreMisura' when VARIABILE is missing, drop non-positive/NaN.
    5) Apply IQR-based outlier filtering.
    6) Aggregate to one record per animal-day (first after stable sort).
    7) Save Parquet output.
    """
    # CF IDs universe
    ids_cf = pd.read_parquet(CF_IDS_PARQUET)["id"].unique()
    log.info("Unique IDs from functional check: %d", len(ids_cf))

    # Load raw EC data and base filters
    df = pd.read_csv(
        RAW_CE_CSV,
        low_memory=False,
        usecols=lambda c: c not in {"idMisuraPrimaria", "siglaProvincia", "codiceRazzaAIA", "codiceSpecieAIA"},
    )
    df = df.query("anno > 2018").drop_duplicates()
    df = df[df["idAnimale"].isin(ids_cf)]
    log.info("[1] Data loaded: %s rows, %s columns", f"{df.shape[0]:,}", df.shape[1])

    # Stable temporal sort and dtype normalization
    df = df.sort_values(["idAnimale", "anno", "mese", "giorno"], kind="mergesort")
    df["idAnimale"] = df["idAnimale"].astype("int64")
    df["giorno"] = df["giorno"].astype("int64")
    df["mese"] = df["mese"].astype("int64")
    df["anno"] = df["anno"].astype("int64")

    # Prefer explicit EC column; fallback to 'valoreMisura' when needed
    if "valoreMisura" in df.columns:
        df[VARIABILE] = df[VARIABILE].fillna(df["valoreMisura"])

    # Keep positive, non-missing EC values
    df = df[df[VARIABILE].notna() & (df[VARIABILE] > 0)]

    # Outlier removal
    df = IQR_filtering(df, VARIABILE)

    # Aggregate to one EC per animal-day
    df_agg = df.groupby(
        ["idAnimale", "anno", "mese", "giorno"], observed=True, as_index=False
    ).agg({VARIABILE: "first"})

    # Free memory
    del df
    gc.collect()

    log.info("Aggregation completed – final rows: %s, columns: %s", f"{df_agg.shape[0]:,}", df_agg.shape[1])

    # Save
    df_agg = df_agg.rename(columns=RENAME_MAP)
    df_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s (%d rows)", OUTPUT_PARQUET, len(df_agg))

    # Cleanup
    del df_agg
    gc.collect()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    ec_main()
