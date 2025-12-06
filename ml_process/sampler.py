"""
Balanced undersampling for mastitis dataset construction.

This module:
- Validates 3-step monthly temporal sequences for biological plausibility.
- Selects first-onset mastitis cases as positives.
- Performs stratified undersampling of healthy animals as negatives, matching the
  strata distribution (age group × lactation phase) of positives.
- Shuffles and saves the balanced dataset to Parquet.

Notes:
- Temporal validity allows month rollovers (e.g., Dec -> Jan) and relaxes month gaps during peripartum/early-lactation windows (≤3 months instead of ≤1).
- Negatives are sampled without replacement and capped by the number of positives in each stratum to avoid class leakage and preserve structure.
"""

from libraries import np, resample, pd, logging, gc, Path


def check_temporal_sequence_vectorized(df: pd.DataFrame) -> np.ndarray:
  
    """
    Validate temporal consistency for (t, t-1, t-2) monthly windows.

    A sequence is valid if:
    - Month differences (with 12-month rollover) are > 0 and ≤ max allowed gap:
       - default ≤ 1;
       - ≤ 3 if 'lactation_phase_peripartum' is active for (t vs t-1);
       - ≤ 3 if 'lactation_phase_early_lactation' is active for (t-1 vs t-2).
    - Years are coherent with potential month rollovers (e.g., Jan(t) vs Dec(t-1)).

    Parameters:
  
    df : pd.DataFrame
        - Must contain columns: ['month', 'month_t-1', 'month_t-2', 'year', 'year_t-1', 'year_t-2', 'lactation_phase_peripartum', 'lactation_phase_early_lactation'].

    Returns:
    
    - np.ndarray --> Boolean mask (length = len(df)) indicating valid sequences.
    """
  
    m0 = df['month'].to_numpy()
    m1 = df['month_t-1'].to_numpy()
    m2 = df['month_t-2'].to_numpy()
    y0 = df['year'].to_numpy()
    y1 = df['year_t-1'].to_numpy()
    y2 = df['year_t-2'].to_numpy()
    peripartum = df['lactation_phase_peripartum'].to_numpy()
    early_lactation = df['lactation_phase_early_lactation'].to_numpy()
    # Month differences with rollover (e.g., Jan(1) - Dec(12) -> (1-12) % 12 = 1)
    delta1 = (m0 - m1) % 12
    delta2 = (m1 - m2) % 12
    # Default max gaps (≤1), relaxed to ≤3 for specific phases
    max_d1 = np.where(peripartum == 1, 3, 1)
    max_d2 = np.where(early_lactation == 1, 3, 1)
    # Month validity (positive forward movement and within allowed gap)
    valid_month = (delta1 > 0) & (delta1 <= max_d1) & (delta2 > 0) & (delta2 <= max_d2)
    # Year coherence allowing rollover between consecutive steps
    year_match_01 = ((m0 >= m1) & (y0 == y1)) | ((m0 < m1) & (y0 == y1 + 1))
    year_match_12 = ((m1 >= m2) & (y1 == y2)) | ((m1 < m2) & (y1 == y2 + 1))
    valid_year = year_match_01 & year_match_12
    return valid_month & valid_year


def undersample_balanced(input_path: Path, output_path: Path) -> None:
  
    """
    Build a balanced dataset by undersampling negatives to match positive strata.

    Steps:
    1) Load wide panel and mark temporal validity with `check_temporal_sequence_vectorized`.
    2) Define positives = first-onset mastitis (1,0,0) & not healthy & valid sequence.
    3) Define pure negatives = (0,0,0) & healthy & valid sequence.
    4) Build strata = cartesian tuple of age and one-hot lactation_phase_* columns.
    5) For each positive stratum, sample (without replacement) up to N_neg = N_pos negatives.
    6) Concatenate, drop helper columns, shuffle, and save to Parquet.

    Parameters:
    
    - input_path : str ---> Path to the input Parquet with wide features and lagged targets.
    output_path : str ---> Path where the balanced Parquet will be written.

    Notes:
    
    - Uses a fixed random_state (42) for reproducibility.
    - If a positive stratum lacks negatives, that stratum contributes only positives.
    """
  
    # Load and temporal validity
    df_wide = pd.read_parquet(input_path).copy()
    df_wide['sequenza_valida'] = check_temporal_sequence_vectorized(df_wide)
    # Positives: first-onset mastitis with valid sequence, not flagged healthy
    positivi = df_wide[
        (df_wide['mastitis'] == 1) &
        (df_wide['mastitis_t-1'] == 0) &
        (df_wide['mastitis_t-2'] == 0) &
        (df_wide['healthy'] == 0) &
        (df_wide['sequenza_valida'])
    ].copy()
    # Pure negatives: consistently healthy with valid sequence
    negativi_puri = df_wide[
        (df_wide['mastitis'] == 0) &
        (df_wide['mastitis_t-1'] == 0) &
        (df_wide['mastitis_t-2'] == 0) &
        (df_wide['healthy'] == 1) &
        (df_wide['sequenza_valida'])
    ].copy()
    logging.info("Positives (first-onset, valid): %d", len(positivi))
    logging.info("Pure negatives (healthy, valid): %d", len(negativi_puri))
    # Strata definition: age × lactation_phase_* (one-hot)
    fase_cols = [c for c in df_wide.columns if c.startswith("lactation_phase_")]
    age_cols = [c for c in df_wide.columns if c.startswith("age")]  # auto-detected
    if not age_cols or not fase_cols:
        missing = []
        if not age_cols:
            missing.append("age")
        if not fase_cols:
            missing.append("lactation_phase_*")
        raise KeyError(f"Required one-hot columns missing for strata: {', '.join(missing)}")
    # Tuple strata to preserve full combinatorial structure
    negativi_puri['strato'] = list(zip(*[negativi_puri[col] for col in age_cols + fase_cols]))
    positivi['strato'] = list(zip(*[positivi[col] for col in age_cols + fase_cols]))
    # Count positives per stratum
    conteggio_strati = positivi['strato'].value_counts().to_dict()
    logging.info("Positive strata: %d unique, total positives: %d", len(conteggio_strati), len(positivi))
    # Stratified undersampling of negatives
    campioni_negativi = []
    for strato, n_pos in conteggio_strati.items():
        subset_neg = negativi_puri[negativi_puri['strato'] == strato]
        if subset_neg.empty:
            # No negatives available for this stratum; keep only positives here
            continue
        campione = resample(subset_neg,replace=False,n_samples=min(len(subset_neg), n_pos),random_state=42)
        campioni_negativi.append(campione)
    if campioni_negativi:
        negativi_finali = pd.concat(campioni_negativi, axis=0, ignore_index=True)
    else:
        logging.warning("No negative samples selected — check strata overlap.")
        negativi_finali = negativi_puri.iloc[0:0].copy()  # empty with same schema
    logging.info("Selected negatives (undersampled): %d", len(negativi_finali))
    # Combine, clean helpers, shuffle for downstream modeling
    df_bilanciato = pd.concat([positivi, negativi_finali], axis=0, ignore_index=True)
    df_bilanciato = df_bilanciato.drop(columns=['strato', 'sequenza_valida'], errors='ignore')
    df_bilanciato = df_bilanciato.sample(frac=1, random_state=42).reset_index(drop=True)
    # Save
    logging.info("Saving balanced dataset to %s …", output_path)
    df_bilanciato.to_parquet(output_path, index=False)
    logging.info(
        "Undersampling completed: total=%d  positives=%d  negatives=%d",
        len(df_bilanciato),
        int(df_bilanciato['mastitis'].sum()),
        int((df_bilanciato['mastitis'] == 0).sum()))
    # Cleanup
    del df_bilanciato, df_wide, positivi, negativi_puri
    if 'negativi_finali' in locals():
        del negativi_finali
    del conteggio_strati, campioni_negativi, age_cols, fase_cols
    gc.collect()






