"""
Pre-processing and domain-informed transformations for the unified dairy dataset.

This module provides two steps:
1) pre_processing: row-level cleaning.
   - Imputes missing breed and birth_date per animal (first non-null).
   - Derives 'age' (years) variable and filter to keep those animals with 2 <= age <= 6
   - Trims records to start from the first calving per animal.
   - Imputes missing breed and birth_date per animal (first non-null).
   - Enforces calving block coherence for 'born'/'nborn' within lactations.

2) dit (Domain-Informed Transformer):
   - Ensures datetime consistency and chooses a 'reference_date' (cf_date > calving_date > t_date).
   - Builds 'healthy' label (absence of diagnosis & SCS <= 5).
   - Derives 'season', and 'lactation_phase' (months since last calving).
   - Flags mastitis events via diagnosis/date logic with 30-day windows.

Outputs:
- Both functions save a Parquet file with the transformed table and log shape summaries.

Notes:
- 'First calving start' is enforced to avoid pre-parity noise.
- Breed/birth_date imputations are per-animal first-observation fills.
- Mastitis heuristic prioritizes CF date. It falls back to calving date or missing-date diagnosis.
"""

from libraries import pd, np, log, gc, Path

### PRE-PROCESSING ###

def pre_processing(input_path: Path, output_path: Path) -> None:
   
    """
    Pre-filter and harmonize records prior to feature engineering.

    Steps:
    
    1) Keep only animals with at least one valid 'birth_date'; impute missing per animal.
    2) Define the variable 'age' (years) as floor((reference_date - birth_date) / 365). Filter out rows with age ≤ 1 and >= 7.
    3) Sort by ['id','year','month'] with a stable order; compute row_number per animal.
    4) Keep rows from the first observed calving (calving == 1) onward.
    5) Fill missing 'calving' with 0 (int).
    6) Impute 'breed' per animal using the first non-null value.
    7) Keep only animals with at least one valid 'birth_date'; impute missing per animal.
    8) Enforce calving-block coherence for 'born'/'nborn':
       - Build cumulative calving counter within animal.
       - For each (animal, block) pair, propagate unique 'born'/'nborn' values.

    Parameters:
    
    - input_path : Path ---> Path to the input Parquet file.
    - output_path : Path ---> Path where the cleaned Parquet file will be saved.

    Returns None
    """
   
    log.info("Starting pre-filtering from file: %s", input_path)
    df = pd.read_parquet(input_path)
    # Keep only animals with at least one valid birth_date 
    id_con_birth = df.loc[df["birth_date"].notna(), "id"].unique()
    df = df[df["id"].isin(id_con_birth)].copy()
    # Birth_date imputation (first non-null per animal)
    birth_per_id = (df.loc[df["birth_date"].notna()].groupby("id", observed=True)["birth_date"].first())
    df["birth_date"] = df["birth_date"].fillna(df["id"].map(birth_per_id))
    # Unified reference date (prefer cf_date, then calving_date, then t_date)
    reference_date = df["cf_date"].combine_first(df["calving_date"]).combine_first(df["t_date"])
    # AGE (years) 
    df["age"] = ((reference_date - df["birth_date"]).dt.days // 365).clip(lower=0).astype(int)
    df = df[(df["age"] > 1) & (df["age"] < 7)] # Remove rows with implausible age to have a functional check ( < 1 years) and those with 7-years bovine (10 rows)
    # Pre-calving filtering 
    # Stable temporal sort (mergesort preserves order on ties)
    df.sort_values(["id", "year", "month"], kind="mergesort", inplace=True)
    # Row index within each animal's timeline
    df["row_number"] = df.groupby("id").cumcount()
    # First observed calving row per animal (if any)
    primo_parto_riga = (df[df["calving"] == 1].groupby("id", observed=True)["row_number"].min())
    # Keep rows from first calving onwards (animals without calving are dropped)
    df = df[df["row_number"] >= df["id"].map(primo_parto_riga)].copy()
    # Calving: fill NaN -> 0 (integer)
    df["calving"] = df["calving"].fillna(0).astype(int)
    # Breed imputation (first non-null per animal) 
    breed_per_id = (df.loc[df["breed"].notna()].groupby("id", observed=True)["breed"].first())
    df["breed"] = df["breed"].fillna(df["id"].map(breed_per_id))
    # Keep only animals with at least one valid birth_date 
    id_con_birth = df.loc[df["birth_date"].notna(), "id"].unique()
    df = df[df["id"].isin(id_con_birth)].copy()
    # Birth_date imputation (first non-null per animal)
    birth_per_id = (df.loc[df["birth_date"].notna()].groupby("id", observed=True)["birth_date"].first())
    df["birth_date"] = df["birth_date"].fillna(df["id"].map(birth_per_id))
    # Calving coherence: propagate born/nborn within calving blocks 
    df["parto_blocco"] = df.groupby("id", observed=True)["calving"].cumsum().astype(int)
    df["chiave_blocco"] = list(zip(df["id"], df["parto_blocco"]))
    mask_calving = df["calving"] == 1
    blocchi_valori = (df.loc[mask_calving, ["chiave_blocco", "born", "nborn"]].drop_duplicates("chiave_blocco"))
    # Create maps (block -> values)
    born_map = dict(zip(blocchi_valori["chiave_blocco"], blocchi_valori["born"]))
    nborn_map = dict(zip(blocchi_valori["chiave_blocco"], blocchi_valori["nborn"]))
    # Assign to all rows in the same calving block
    df["born"] = df["chiave_blocco"].map(born_map)
    df["nborn"] = df["chiave_blocco"].map(nborn_map)
    # Clean up helper columns 
    df.drop(columns=["chiave_blocco"], inplace=True)
    df.drop(columns=["parto_blocco", "row_number"], errors="ignore", inplace=True)
    df.reset_index(drop=True, inplace=True)
    gc.collect()
    # Save
    df.to_parquet(output_path, index=False)
    log.info("File saved: %s (%d rows, %d columns)", output_path, len(df), df.shape[1])
    # Cleanup references for memory hygiene (optional)
    del df, born_map, nborn_map, blocchi_valori, mask_calving, primo_parto_riga, breed_per_id, id_con_birth, reference_date
    gc.collect()
    log.info("Pre-processing completed.")



### DOMAIN-INFORMED TRANSFORMER ###

def dit(input_path: Path, output_path: Path) -> None:
   
    """
    Apply domain-informed transformations to the cleaned dataset.

    Features & Labels:
    
    - 'healthy': 1 if no diagnosis AND SCS <= 5; 0 otherwise.
    - 'season': based on 'month' (winter/spring/summer/autumn).
    - 'lactation_phase': derived from months since last calving.
    - 'mastitis': diagnosis + temporal logic with 30-day windows,
      prioritizing CF date over calving date when both exist.

    Steps:
    
    1) Load data.
    2) Coerce datetime columns ['birth_date','cf_date','t_date','calving_date'].
    4) Compute 'healthy' using diagnosis & SCS thresholds.
    5) Derive 'season' and 'lactation_phase'.
    6) Compute 'mastitis' with diagnosis/date rules and ≤30-day windows.
    7) Save.

    Parameters:
   
    - input_path : Path ---> Path to the input Parquet file produced by pre_processing().
    - output_path : Path ---> Path where the transformed Parquet file will be saved.

    Returns None
    """
   
    # Load dataset
    df = pd.read_parquet(input_path)
    # Remove rows with missing breed (downstream models may expect it)
    df = df[df["breed"].notna()]
    # Ensure datetime consistency
    for col in ["birth_date", "cf_date", "t_date", "calving_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # HEALTHY label 
    ids_with_diagnosis = df.loc[df["diagnosis"].notna(), "id"].unique()
    df["scs"] = pd.to_numeric(df["scs"], errors="coerce")
    ids_with_high_scs = df.loc[df["scs"] > 5, "id"].unique()
    not_healthy_ids = set(ids_with_diagnosis).union(set(ids_with_high_scs))
    df["healthy"] = (~df["id"].isin(not_healthy_ids)).astype(int)
    # SEASON 
    season_conditions = [
        df["month"].isin([12, 1, 2]),
        df["month"].isin([3, 4, 5]),
        df["month"].isin([6, 7, 8]),
        df["month"].isin([9, 10, 11]),
    ]
    season_labels = ["winter", "spring", "summer", "autumn"]
    df["season"] = np.select(season_conditions, season_labels, default=pd.NA)
    # LACTATION PHASE 
    # months_since_calving relative to the record month (1st day convention)
    df = df.sort_values(["id", "year", "month"], kind="mergesort")
    df["calving_date"] = pd.to_datetime(df["calving_date"])
    df = df.sort_values(["id", "year", "month"], kind="mergesort")
    df["last_calving_date"] = df.groupby("id")["calving_date"].ffill()
    df["record_date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["months_since_calving"] = ((df["record_date"].dt.year - df["last_calving_date"].dt.year) * 12 + (df["record_date"].dt.month - df["last_calving_date"].dt.month))
    df.loc[df["last_calving_date"].isna(), "months_since_calving"] = pd.NA
    lactation_conditions = [
        df["months_since_calving"] == 0,
        df["months_since_calving"] == 1,
        df["months_since_calving"].isin([2, 3]),
        df["months_since_calving"].isin([4, 5, 6]),
        df["months_since_calving"] >= 7,
    ]
    lactation_labels = ["peripartum", "early_lactation", "peak", "mid_lactation", "late_lactation"]
    df["lactation_phase"] = np.select(lactation_conditions, lactation_labels, default=pd.NA)
    # Drop helper columns
    df.drop(columns=["record_date", "last_calving_date"], inplace=True)
    # MASTITIS label 
    # Diagnosis + acceptable temporal windows 
    diagnosis_no_dates = (
        df["diagnosis"].notna()
        & df["cf_date"].isna()
        & df["calving_date"].isna()
    )
    cf_within_30_days = (
        df["cf_date"].notna()
        & df["t_date"].notna()
        & (df["cf_date"] <= df["t_date"])
        & ((df["t_date"] - df["cf_date"]).dt.days <= 30)
    )
    calving_within_30_days = (
        df["cf_date"].isna()
        & df["calving_date"].notna()
        & df["t_date"].notna()
        & (df["calving_date"] <= df["t_date"])
        & ((df["t_date"] - df["calving_date"]).dt.days <= 30)
    )
    mastitis_mask_updated = (diagnosis_no_dates | cf_within_30_days | calving_within_30_days)
    df["mastitis"] = mastitis_mask_updated.fillna(False).astype(int)
    # Drop helper columns to reduce memory footprint
    df.drop(columns=["months_since_calving"], inplace=True)
    # Save result (require non-missing age)
    df = df.dropna(subset=["age"])
    df.to_parquet(output_path, index=False)
    log.info("File saved: %s (%d rows, %d columns)", output_path, len(df), df.shape[1])
    # Cleanup references to help GC (optional)
    del df, not_healthy_ids, ids_with_diagnosis, ids_with_high_scs
    gc.collect()
    log.info("Domain-Informed transformation completed.")







