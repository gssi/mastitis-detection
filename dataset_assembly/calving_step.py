"""
Calving events pipeline.

This script:
- Loads raw calving records from CSV.
- Keeps only animals present in the functional check (CF) dataset.
- Aggregates births per animal-day and collapses to at most one calving per month.
- Applies plausibility filters (e.g., keep born counts < 3, remove empty events).
- Logs per-animal calving counts before/after preprocessing for coherence checks.
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- Ensures calving events are deduplicated and plausible for downstream analyses.

Outputs:
- parti_agg.parquet: cleaned calving events with totals of live/dead births and a binary flag.
"""

from libraries import pd, Path, log, gc

# =========================
# PATHS AND STATICS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "db" / "parto" / "parto.csv"
CF_IDS_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "cf_ids.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "parti_agg.parquet"

# Column rename map (Italian -> English)
RENAME_MAP = {
    "idAnimale": "id",
    "giorno": "day",
    "mese": "month",
    "anno": "year",
    "NatiVivi": "born",
    "NatiMorti": "nborn",
    "data": "calving_date",
    "Parto": "calving",
}


# =========================
# FUNCTIONS
# =========================

def count_parts(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Compute per-animal calving event counts and log their distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'idAnimale' (per-animal rows).
    label : str
        A label to distinguish stages (e.g., 'raw', 'final') in logs.

    Returns
    -------
    pd.Series
        Series indexed by 'idAnimale' with the number of events per animal.
    """
    counts = df.groupby("idAnimale", observed=True).size()
    # Cast to int for safe %d formatting in logs (median could be float)
    cmin = int(counts.min()) if len(counts) else 0
    cmed = int(counts.median()) if len(counts) else 0
    cmax = int(counts.max()) if len(counts) else 0
    log.info("Parts distributions[%s] – min:%d  median:%d  max:%d", label, cmin, cmed, cmax)
    return counts


# =========================
# MAIN
# =========================

def calving_main() -> None:
    """
    End-to-end processing of calving events:
    1) Load CF IDs.
    2) Load raw calving CSV; keep years > 2018 and deduplicate.
    3) Filter to CF animals, sort temporally.
    4) Aggregate per animal-day (sum male/female, live/dead).
    5) Filter implausible counts (< 3 per category).
    6) Keep at most one calving per animal-month (first day in month).
    7) Compute totals (live/dead), build datetime, drop empty events.
    8) Log coherence of counts before/after preprocessing.
    9) Save cleaned dataset to Parquet.
    """
    # Load CF IDs universe
    ids_cf = pd.read_parquet(CF_IDS_PARQUET)["id"].unique()
    log.info("Unique IDs from functional check: %d", len(ids_cf))

    # Load raw calving data and base filters
    df = pd.read_csv(
        RAW_CSV,
        low_memory=False,
        usecols=lambda c: c not in {"idMisuraPrimaria", "codiceRazzaAIA", "codiceSpecieAIA", "siglaProvincia"},
    )
    df = df.query("anno > 2018").drop_duplicates()

    # Keep only animals present in CF universe
    df = df[df["idAnimale"].isin(ids_cf)]

    # Keep a copy for pre/post event-count comparison
    df_orig = df.copy()

    # Stable temporal sort (mergesort preserves order on ties)
    df = df.sort_values(["idAnimale", "anno", "mese", "giorno"], kind="mergesort")

    # Aggregate to per-animal-day totals (sex and vitality)
    df2 = df.groupby(
        ["idAnimale", "giorno", "mese", "anno"], observed=True, as_index=False
    ).agg({
        "NumeroFemmineNateVive": "sum",
        "NumeroFemmineNateMorte": "sum",
        "NumeroMaschiNatiVivi": "sum",
        "NumeroMaschiNatiMorti": "sum",
    })

    # Free memory
    del df
    gc.collect()

    # Plausibility filter: each category count must be < 3
    for col in ["NumeroFemmineNateVive", "NumeroFemmineNateMorte", "NumeroMaschiNatiVivi", "NumeroMaschiNatiMorti"]:
        df2 = df2[df2[col] < 3]
    log.info("After filtering for per-category born < 3: %d animals", df2["idAnimale"].nunique())

    # Collapse to at most one calving per animal-month (keep earliest day)
    df2["mese_anno"] = df2["anno"].astype(str) + "-" + df2["mese"].astype(str).str.zfill(2)
    df2 = (
        df2.sort_values(["idAnimale", "mese_anno", "giorno"], kind="mergesort")
           .drop_duplicates(subset=["idAnimale", "mese_anno"], keep="first")
           .drop(columns="mese_anno")
    )

    # Totals (live/dead) and event date
    df2["NatiVivi"] = df2["NumeroFemmineNateVive"] + df2["NumeroMaschiNatiVivi"]
    df2["NatiMorti"] = df2["NumeroFemmineNateMorte"] + df2["NumeroMaschiNatiMorti"]
    df2["data"] = pd.to_datetime(
        {"year": df2["anno"], "month": df2["mese"], "day": df2["giorno"]},
        errors="coerce",
    )

    # Binary event flag
    df2["Parto"] = 1

    # Remove raw component columns now represented by totals
    df2 = df2.drop(
        columns=[
            "NumeroFemmineNateMorte",
            "NumeroFemmineNateVive",
            "NumeroMaschiNatiVivi",
            "NumeroMaschiNatiMorti",
        ]
    )

    # Remove empty events (no live nor dead births)
    df2 = df2[~((df2["NatiVivi"] == 0) & (df2["NatiMorti"] == 0))]
    log.info("After removing empty calving events: %d animals", df2["idAnimale"].nunique())

    # Coherence check: per-animal event counts (raw vs final)
    parts_start = count_parts(df_orig, "raw")
    del df_orig
    gc.collect()

    parts_end = count_parts(df2, "final")
    diff = (parts_start - parts_end).fillna(0).astype(int)
    mismatch_ids = diff[diff != 0].index.tolist()
    if mismatch_ids:
        log.warning("%d animals have gained/lost calving events after preprocessing.", len(mismatch_ids))
        log.debug("IDs with mismatch: %s", mismatch_ids)
    else:
        log.info("Calving events count is coherent for all animals.")

    # Save
    df2 = df2.rename(columns=RENAME_MAP)
    df2.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s  (%d rows)", OUTPUT_PARQUET, len(df2))

    # Cleanup
    del df2
    gc.collect()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    calving_main()

