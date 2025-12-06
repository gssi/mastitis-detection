"""
Merged dataset builder.

This script:
- Safely merges multiple preprocessed datasets (cf_agg, ltts_agg, ce_agg, parti_agg, coppie_trat_agg, ana_agg).
- Uses a robust multi-DataFrame join helper ('unisci') that harmonizes key dtypes before merging.
- Produces a single, analysis-ready Parquet file.

Join plan:
1) LEFT join:  cf_agg <- ltts_agg <- ce_agg   on [id, day, month, year]
2) OUTER join: (step1 without 'day') <-> parti_agg <-> coppie_trat_agg    on [id, month, year]
3) INNER join: (step2) x ana_agg   on [id]  

Output:
- merged_dataset.parquet
"""

from libraries import Path, log, pd, gc, np, reduce


### PATHS AND STATICS ###

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PARQUET = (
    PROJECT_ROOT
    / "mammary_diseases_indicators"
    / "temporary_datasets"
    / "merged_dataset.parquet"
)


### HELPERS ###

def unisci(lista_df, lista_chiavi, metodo):
    
    """
    Safely merge multiple DataFrames on given keys, harmonizing dtypes. For each key in 'lista_chiavi':
    
    - If any dtype is 'category' or any is 'object'/'string' -> cast all to str.
    - If all are numeric -> promote to a common numeric dtype (np.promote_types).
    - Otherwise raise TypeError.
    After harmonization, perform a chained merge via functools.reduce.

    PARAMETERS:
    
    - lista_df : list[pd.DataFrame] ---> DataFrames to merge.
    - lista_chiavi : list[str] ---> Column names to merge on (must exist in all DataFrames).
    - metodo : {'inner','left','right','outer'} ---> Merge strategy passed to pd.merge.

    RETURNS:
    
    - pd.DataFrame ---> Merged DataFrame.

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
        # Categorical or string-like -> cast everything to string
        if any(t.name == "category" for t in tipi) or ("object" in tipo_names or "string" in tipo_names):
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(str)
            continue
        # All numeric -> promote to a common numeric dtype
        if all(np.issubdtype(t, np.number) for t in tipi):
            tipo_comune = tipi[0]
            for t in tipi[1:]:
                tipo_comune = np.promote_types(tipo_comune, t)
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(tipo_comune)
            continue
        # Mixed / unsupported types
        raise TypeError(f"Not compatible types for '{chiave}': {tipo_names}")
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=lista_chiavi, how=metodo), lista_df)
    if merged_df.empty:
        raise ValueError("Join step created an empty DataFrame: please, check.")
    return merged_df


### MAIN ###

def merge_main() -> None:
    
    """
    Build the final merged dataset by following the 3-step join plan.

    Steps:
    1) LEFT join cf_agg, ltts_agg, ce_agg on [id, day, month, year].
    2) OUTER join with parti_agg and coppie_trat_agg on [id, month, year] (drop 'day').
    3) INNER join with ana_agg on [id].
    4) Save Parquet output and free memory.
    """
    
    log.info("START MERGING PHASE...")
    # Step 1: LEFT join on daily keys
    log.info("Step 1 – LEFT join between cf_agg, ltts_agg, ce_agg on keys [id, day, month, year]")
    step1 = unisci([
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "cf_agg.parquet"),
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "ltts_agg.parquet"),
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "ce_agg.parquet"),
    ], ["id", "day", "month", "year"], "left")
    gc.collect()
    # Step 2: OUTER join on monthly keys (drop 'day')
    log.info("Step 2 – OUTER join with parti_agg and coppie_trat_agg on keys [id, month, year]")
    step2 = unisci([
        step1.drop(columns="day", errors="ignore"),
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "parti_agg.parquet").drop(columns="day", errors="ignore"),
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "coppie_trat_agg.parquet").drop(columns="day", errors="ignore"),
    ], ["id", "month", "year"], "outer")
    del step1
    gc.collect()
    # Step 3: INNER join with anagraphic info
    log.info("Step 3 – INNER join with ana_agg on keys [id]")
    merged = unisci([
        step2,
        pd.read_parquet(PROJECT_ROOT / "mammary_diseases_indicators" / "temporary_datasets" / "ana_agg.parquet"),
    ], ["id"], "inner")
    del step2
    gc.collect()
    # Save
    merged = merged.drop("Marca", axis=1, errors="ignore") # Drop 'Marca' column to keep only animal ID
    merged.to_parquet(OUTPUT_PARQUET, index=False)
    log.info(
        "Merged dataset saved ➔ %s (%d rows, %d columns)",
        OUTPUT_PARQUET, len(merged), merged.shape[1])
    # Cleanup
    del merged
    gc.collect()
    log.info("END MERGING PHASE")

### ENTRY POINT ###

if __name__ == "__main__":
    merge_main()






