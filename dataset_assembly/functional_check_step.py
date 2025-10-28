"""
Data pipeline for Control Function (Controllo Funzionale) records.

This script:
- Loads raw CF data from CSV.
- Normalizes and reconciles breed labels per animal over time.
- Aggregates daily measurements per animal.
- Applies basic quality filters (positivity checks and IQR-based outlier removal).
- Saves a compact Parquet dataset and the list of unique animal IDs.

Context:
- Designed for mammary disease indicators workflow.
- Prioritizes clarity, reproducibility, and memory-conscious steps.

Outputs:
- cf_agg.parquet: daily aggregated and filtered CF measures
- cf_ids.parquet: unique animal IDs present after filtering

Notes for reviewers:
- Breed coherence is enforced by forward/backward fill within animal and a final majority-vote pass.
- Outlier filtering uses a standard Tukey IQR rule (configurable multiplier).
"""

from libraries import pd, Path, log, gc, Counter


### PATHS AND STATICS ###

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CF_CSV = PROJECT_ROOT / "db" / "controllo_funzionale" / "controllo_funzionale.csv"
OUTPUT_PARQUET = (
    PROJECT_ROOT
    / "mammary_diseases_indicators"
    / "temporary_datasets"
    / "cf_agg.parquet"
)

# Column rename map (Italian -> English)
RENAME_MAP = {
    "idAnimale": "id",
    "razza": "breed",
    "giorno": "day",
    "mese": "month",
    "anno": "year",
    "LatteAlle24Ore": "milk",
    "ProteineAlle24Ore": "protein",
    "GrassoAlle24Ore": "fat",
    "LinearScore": "scs",
    "DataControlloFunzionaleLatte": "cf_date",
}

# Mapping of heterogeneous breed codes to canonical Italian labels
MAPPATURA_RAZZE = {
    "00": "meticcia", "0": "meticcia", "0.0": "meticcia",
    "01": "bruna",    "1": "bruna",    "1.0": "bruna",
    "02": "frisona",  "2": "frisona",  "2.0": "frisona",
    "04": "pezzata",  "4": "pezzata",  "4.0": "pezzata",
}

# Keep only these breeds in the final dataset
RAZZE_DI_INTERESSE = ["frisona", "pezzata", "meticcia", "bruna"]


### HELPERS ###

def map_razza(series: pd.Series) -> pd.Series:
    
    """
    Map heterogeneous breed codes to a categorical canonical label.

    Parameters
    ----------
    series : pd.Series
        Input series with breed codes (e.g., 'codiceRazzaAIA') as strings.

    Returns
    -------
    pd.Series
        Categorical series of mapped breed labels; unmapped codes become NaN.
    """
    
    return pd.Categorical(series.map(MAPPATURA_RAZZE))


def fix_breed_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Enforce per-animal breed coherence by majority vote across records.

    For each animal (idAnimale), select the most frequent observed 'razza'
    and assign it to all its rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['idAnimale', 'razza'].

    Returns
    -------
    pd.DataFrame
        DataFrame with 'razza' coerced to the per-animal mode.
    """
    
    log.info("Solving breed conflicts by per-animal majority vote...")
    mode_map = (
        df.groupby("idAnimale", observed=True)["razza"]
          .agg(lambda x: Counter(x).most_common(1)[0][0])
    )
    df["razza"] = df["idAnimale"].map(mode_map)
    return df


def fill_missing_breed(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Fill missing breed labels within each animal over time and resolve conflicts.

    Steps:
    1) Sort by animal and date components to obtain temporal order.
    2) Forward-fill and backward-fill 'razza' within each animal.
    3) Apply a final per-animal majority-vote to resolve residual inconsistencies.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['idAnimale', 'anno', 'mese', 'razza'].

    Returns
    -------
    pd.DataFrame
        Same columns, with 'razza' filled/coerced for coherence.
    """
    
    # Stable sort keeps earlier ordering when equal keys (useful if day is absent at this step)
    df = df.sort_values(["idAnimale", "anno", "mese"], kind="mergesort")
    # Intra-animal propagation of known labels
    df["razza"] = df.groupby("idAnimale", observed=True)["razza"].ffill()
    df["razza"] = df.groupby("idAnimale", observed=True)["razza"].bfill()
    log.info("Ensuring breed coherence within each animal over time...")
    df = fix_breed_conflicts(df)
    log.info("Breed coherence applied.")
    return df


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


### MAIN FUNCTION ###

def cf_main() -> None:
    
    """
    End-to-end pipeline:
    - Load CSV and retain relevant years/columns.
    - Normalize breeds and keep a set of target breeds.
    - Aggregate per animal-day using 'first' (assumes prior dedup or stable CF collection).
    - Build a proper datetime column.
    - Apply positivity and IQR-based outlier filters.
    - Save final Parquet outputs and unique animal IDs.
    """
    
    # Load & base filtering
    cf = pd.read_csv(
        RAW_CF_CSV,
        low_memory=False,
        # Drop large/irrelevant columns early to save memory
        usecols=lambda c: c not in {"idMisuraPrimaria", "siglaProvincia"},
        dtype={"codiceRazzaAIA": "string", "codiceSpecieAIA": "string"},
    )
    # Keep recent years only and drop exact duplicates
    cf = cf.query("anno > 2018").drop_duplicates()
    # Normalize breeds from raw code -> canonical label
    cf["razza"] = map_razza(cf["codiceRazzaAIA"])
    # Fill missing labels and enforce consistency across an animal's history
    cf = fill_missing_breed(cf)
    # Restrict to breeds of interest
    cf = cf[cf["razza"].isin(RAZZE_DI_INTERESSE)]
    # Aggregate to one record per animal-day
    group_cols = ["idAnimale", "razza", "anno", "mese", "giorno"]
    agg_dict = {
        "LinearScore": "first",
        "LatteAlle24Ore": "first",
        "ProteineAlle24Ore": "first",
        "GrassoAlle24Ore": "first",
    }
    cf_agg = cf.groupby(group_cols, observed=True, as_index=False).agg(agg_dict)
    # Free memory (cf no longer needed)
    del cf
    gc.collect()
    # Construct a true datetime index for CF date; coerce invalid dates to NaT
    cf_agg["DataControlloFunzionaleLatte"] = pd.to_datetime(
        {"year": cf_agg["anno"], "month": cf_agg["mese"], "day": cf_agg["giorno"]},
        errors="coerce",
    )
    # Quality filters
    # Positivity checks for production components; SCS can be zero
    for v in ["LatteAlle24Ore", "GrassoAlle24Ore", "ProteineAlle24Ore", "LinearScore"]:
        if v != "LinearScore":
            cf_agg = cf_agg[cf_agg[v] > 0]
        cf_agg = IQR_filtering(cf_agg, v)
    # Save main dataset
    cf_agg = cf_agg.rename(columns=RENAME_MAP)
    cf_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s  (%d rows)", OUTPUT_PARQUET, len(cf_agg))
    # Save unique IDs (post-filtering universe)
    unique_ids = cf_agg["id"].drop_duplicates()
    OUTPUT_IDS_PARQUET = (
        PROJECT_ROOT
        / "mammary_diseases_indicators"
        / "temporary_datasets"
        / "cf_ids.parquet"
    )
    unique_ids.to_frame().to_parquet(OUTPUT_IDS_PARQUET, index=False)
    log.info("Unique IDs saved ➔ %s  (%d ID)", OUTPUT_IDS_PARQUET, len(unique_ids))
    # Cleanup
    del cf_agg
    gc.collect()


### ENTRY POINT ###

if __name__ == "__main__":
    cf_main()



