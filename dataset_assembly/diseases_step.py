"""
Treatments (mastitis) pipeline.

This script:
- Loads raw treatment records from CSV.
- Keeps only rows for a target diagnosis (mastitis).
- Parses dates, filters implausible/old years, and deduplicates to one treatment per animal-month.
- Joins eartag codes to internal animal IDs (via 'coppie' mapping).
- Restricts to animals present in the functional check (CF) universe.
- Saves a compact Parquet dataset for downstream analyses.

Context:
- Part of the mammary diseases indicators workflow.
- Emphasizes traceability: distribution logs at each filtering step and coherent merges.

Outputs:
- trat_agg.parquet: cleaned treatments with parsed date and per-month sampling.
- coppie_trat_agg.parquet: same as above, kept for cross-checks with 'coppie' mapping.
"""

from libraries import pd, Path, log, gc, reduce, np

# =========================
# PATHS AND STATICS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "db" / "trattamenti" / "trattamenti.csv"
COPPIE_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "coppie.parquet"
CF_IDS_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "cf_ids.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "trat_agg.parquet"
COPPIE_TRAT_PARQUET = PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "coppie_trat_agg.parquet"

TARGET_DIAGNOSIS = "MAMMARIE"

# Column rename map (Italian -> English)
RENAME_MAP = {
    "CAPO_IDENTIFICATIVO": "Marca",
    "TIPODIAGNOSI_CODICE": "diagnosis",
    "TRAT_DT_INIZIO_parsed": "t_date",
    "giorno": "day",
    "mese": "month",
    "anno": "year",
}


# =========================
# HELPERS
# =========================

def counts_per_animal(df: pd.DataFrame) -> pd.Series:
    """
    Per-animal event counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'CAPO_IDENTIFICATIVO'.

    Returns
    -------
    pd.Series
        Counts by 'CAPO_IDENTIFICATIVO'.
    """
    return df.groupby("CAPO_IDENTIFICATIVO", observed=True).size()


def show_dist(label: str, s: pd.Series) -> None:
    """
    Log distribution summary for counts series.

    Parameters
    ----------
    label : str
        Label for the log line (e.g., 'Step 0 (raw)').
    s : pd.Series
        Series of counts by animal.
    """
    if len(s) == 0:
        log.info("%s – rows:0  n_ids:0  min:0  median:0  max:0", label)
        return
    # Cast to int for safe %d formatting (median may be float)
    rows = int(s.sum())
    n_ids = int(len(s))
    cmin = int(s.min())
    cmed = int(s.median())
    cmax = int(s.max())
    log.info("%s – rows:%d  n_ids:%d  min:%d  median:%d  max:%d", label, rows, n_ids, cmin, cmed, cmax)


def unisci(lista_df, lista_chiavi, metodo):
    """
    Safely merge multiple DataFrames on given keys, harmonizing dtypes.

    For each key in 'lista_chiavi':
    - If any dtype is 'category', or any is 'object'/'string', cast all to str.
    - If all are numeric, promote to a common numeric dtype with numpy's 'promote_types'.
    - Otherwise, raise TypeError.
    After harmonization, perform a chained merge using functools.reduce.

    Parameters
    ----------
    lista_df : list[pd.DataFrame]
        DataFrames to merge.
    lista_chiavi : list[str]
        Column names to merge on (must exist in all DataFrames).
    metodo : {'inner','left','right','outer'}
        Merge strategy.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.

    Raises
    ------
    KeyError
        If a join key is missing in any DataFrame.
    TypeError
        If key dtypes are incompatible and cannot be harmonized.
    ValueError
        If the resulting merge is empty (likely a join-key mismatch).
    """
    if not lista_df:
        return pd.DataFrame()

    for chiave in lista_chiavi:
        tipi = []
        for df in lista_df:
            if chiave not in df.columns:
                raise KeyError(f"Column '{chiave}' is not in all DataFrames.")
            tipi.append(df[chiave].dtype)

        tipo_names = [t.name for t in tipi]

        # If any categorical OR any string/object -> cast all to string to avoid category code mismatch
        if any(t.name == "category" for t in tipi) or ("object" in tipo_names or "string" in tipo_names):
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(str)
            continue

        # All numeric -> promote to a common supertype (e.g., int64 vs int32 -> int64)
        if all(np.issubdtype(t, np.number) for t in tipi):
            tipo_comune = tipi[0]
            for t in tipi[1:]:
                tipo_comune = np.promote_types(tipo_comune, t)
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(tipo_comune)
            continue

        raise TypeError(f"Not compatible types for '{chiave}': {tipo_names}")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=lista_chiavi, how=metodo), lista_df)
    if merged_df.empty:
        raise ValueError("Join step created an empty dataframe: please, check.")
    return merged_df


# =========================
# MAIN
# =========================

def treat_main(keep_all_years: bool = False) -> None:
    """
    End-to-end processing of mastitis treatments.

    Steps:
    1) Load CF IDs universe.
    2) Load raw treatments, keep only target diagnosis (mastitis).
    3) Parse dates; drop invalid.
    4) Filter by year (>2019 unless keep_all_years=True).
    5) Deduplicate to one treatment per animal-month (keep earliest day).
    6) Join with 'coppie' (eartag -> internal id), then restrict to CF IDs.
    7) Save Parquet outputs.

    Parameters
    ----------
    keep_all_years : bool, default False
        If True, skip the year filter; otherwise keep only year > 2019.
    """
    # -------- CF IDs universe
    ids_cf = pd.read_parquet(CF_IDS_PARQUET)["id"].unique()
    log.info("Unique IDs from functional check: %d", len(ids_cf))

    # -------- Read raw treatments (mastitis only)
    cols = ["CAPO_IDENTIFICATIVO", "TIPODIAGNOSI_CODICE", "TRAT_DT_INIZIO"]
    df = pd.read_csv(
        RAW_CSV,
        low_memory=False,
        usecols=cols,
        dtype={"CAPO_IDENTIFICATIVO": "string", "TIPODIAGNOSI_CODICE": "string"},
    )
    df = df[df["TIPODIAGNOSI_CODICE"] == TARGET_DIAGNOSIS].copy()
    log.info("%d rows related to mastitis diagnosis uploaded", len(df))

    # Remove rows with missing start date
    df = df.dropna(subset=["TRAT_DT_INIZIO"])

    # Distribution before parsing
    c0 = counts_per_animal(df)
    show_dist("Step 0 (raw)", c0)

    # Parse with day-first format; drop unparseable
    df["TRAT_DT_INIZIO_parsed"] = pd.to_datetime(df["TRAT_DT_INIZIO"], errors="coerce", dayfirst=True)
    parse_fail = df["TRAT_DT_INIZIO_parsed"].isna()
    ids_parse = set(df.loc[parse_fail, "CAPO_IDENTIFICATIVO"])
    rows_parse_fail = int(parse_fail.sum())
    df = df[~parse_fail]
    if rows_parse_fail:
        log.info("%d rows with invalid dates removed (involved animals: %d)", rows_parse_fail, len(ids_parse))

    c1 = counts_per_animal(df)
    show_dist("Step 1 (parsed)", c1)

    # Year extraction & filtering
    df["anno"] = df["TRAT_DT_INIZIO_parsed"].dt.year.astype(int)
    if not keep_all_years:
        mask_old = df["anno"] <= 2019
        ids_old = set(df.loc[mask_old, "CAPO_IDENTIFICATIVO"])
        rows_old = int(mask_old.sum())
        df = df[~mask_old]
        if rows_old:
            log.info("%d rows with year ≤ 2019 removed (involved animals: %d)", rows_old, len(ids_old))
    else:
        log.info("Year filter deactivated.")

    c2 = counts_per_animal(df)
    show_dist("Step 2 (year > 2019)", c2)

    # Coherence check (how many events lost by animal)
    delta = (c0 - c2).fillna(0).astype(int)
    mismatch = delta[delta != 0]
    if mismatch.empty:
        log.info("Coherent treatments count for all animals.")
    else:
        log.warning("%d animals with gained/lost treatments after filtering.", len(mismatch))
        log.info("    - %d associated to invalid dates.", len(ids_parse & set(mismatch.index)))
        if not keep_all_years:
            log.info("    - %d associated to year ≤ 2019", len(ids_old & set(mismatch.index)))
        log.debug("Example IDs: %s", list(mismatch.index)[:20])

    # Day and month components for dedup to first treatment in month
    df["giorno"] = df["TRAT_DT_INIZIO_parsed"].dt.day.astype(int)
    df["mese"] = df["TRAT_DT_INIZIO_parsed"].dt.month.astype(int)

    # Sort by date and deduplicate per animal-month (keep earliest)
    df = df.sort_values("TRAT_DT_INIZIO_parsed", kind="mergesort")
    df = df.drop_duplicates(subset=["CAPO_IDENTIFICATIVO", "anno", "mese"], keep="first")
    log.info("After selecting the first date per animal-month: %d rows", len(df))

    # Join with 'coppie' (eartag -> internal id)
    # 'coppie' contains mapping from external 'idAnimale'/'Marca' to internal 'id'
    coppie_df = pd.read_parquet(COPPIE_PARQUET).rename(columns={"idAnimale": "id"})
    df = df.rename(columns={"CAPO_IDENTIFICATIVO": "Marca"})
    df_joined = unisci([coppie_df, df], ["Marca"], "inner")
    log.info("After join: %d rows, %d animals with known ID", len(df_joined), df_joined["id"].nunique())

    # Free memory
    del coppie_df, df
    gc.collect()

    # Keep only animals in CF universe
    df_joined = df_joined[df_joined["id"].isin(ids_cf)]
    log.info("After filtering with CF IDs: %d rows, %d animals", len(df_joined), df_joined["id"].nunique())

    # Save outputs
    df_joined = df_joined.drop("TRAT_DT_INIZIO", axis=1)
    df_joined = df_joined.rename(columns=RENAME_MAP)

    df_joined.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved ➔ %s  (%d rows)", OUTPUT_PARQUET, len(df_joined))

    df_joined.to_parquet(COPPIE_TRAT_PARQUET, index=False)

    # Cleanup
    del df_joined
    gc.collect()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    treat_main()
