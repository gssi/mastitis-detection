"""
Wide-format feature engineering for longitudinal dairy records.

This module:
- Loads a Parquet table with per-animal monthly observations.
- Drops non-predictive/meta columns.
- One-hot encodes categorical features (season, lactation_phase).
- Creates temporal lag features over selected base variables.
- Filters implausible/irrelevant years and removes incomplete rows.
- Saves a compact wide-format Parquet dataset.

Intended use:
- Downstream ML models needing fixed-size, lag-augmented feature vectors.
"""

from libraries import pd, logging, Path

# =========================
# STATIC VARIABLES
# =========================

# Base columns to lag; missing ones are ignored with a warning.
WIDE_COLS = ["scs", "milk", "protein", "fat", "lactose", "ec", "mastitis", "month", "year"]


# =========================
# FUNCTIONS
# =========================

def create_wide(
    input_path: Path,
    output_path: Path,
    lag_steps=None,
    id_col: str = "id",
    time_cols: tuple[str, str] = ("year", "month"),
    min_year: int = 2020,
) -> None:
    """
    Transform a longitudinal dataset into a wide, lag-augmented table.

    Parameters
    ----------
    input_path : str
        Path to input Parquet file.
    output_path : str
        Path where the wide-format Parquet will be written.
    lag_steps : list[int] or None, optional
        List of positive integers representing temporal lags to compute
        within-animal (e.g., [1, 2]). If None, defaults to [1, 2].
    id_col : str, default "id"
        Column identifying the animal/subject.
    time_cols : tuple[str, str], default ("year", "month")
        Tuple with (year_col, month_col) used to order records over time.
    min_year : int, default 2020
        Keep only rows with year_col >= min_year before final NA-drop.

    Notes
    -----
    - Categorical one-hot: 'season', 'lactation_phase' (ignored if absent).
    - Columns silently ignored on drop: ['cf_date', 'calving_date', 'calving',
      'diagnosis', 't_date', 'birth_date', 'breed'].
    - Rows with any NA after lag creation are removed to guarantee complete
      feature vectors for modeling.
    """
    # Defaults & basic validation
    if lag_steps is None:
        lag_steps = [1, 2]
    if not isinstance(lag_steps, (list, tuple)) or not all(isinstance(k, int) and k > 0 for k in lag_steps):
        raise ValueError("lag_steps must be a list/tuple of positive integers.")
    if len(time_cols) != 2:
        raise ValueError("time_cols must be a tuple of two column names: (year_col, month_col).")

    ycol, mcol = time_cols
    logging.info("Starting feature engineering…")

    # Load & initial pruning
    df = pd.read_parquet(input_path)

    # Drop non-predictive/meta columns if present
    drop_cols = ["cf_date", "calving_date", "calving", "diagnosis", "t_date", "birth_date", "breed"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure id/time columns exist
    missing_key_cols = [c for c in [id_col, ycol, mcol] if c not in df.columns]
    if missing_key_cols:
        raise KeyError(f"Missing required columns: {missing_key_cols}")

    # Categorical encoding 
    cat_cols = [c for c in ["season", "lactation_phase"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)

    # Ensure time columns are integers for sorting/lag logic
    df[[ycol, mcol]] = df[[ycol, mcol]].astype(int)

    # Optional: coerce boolean target-like columns to int (e.g., mastitis flags)
    for maybe_bool in ["mastitis"]:
        if maybe_bool in df.columns and df[maybe_bool].dtype == bool:
            df[maybe_bool] = df[maybe_bool].astype(int)

    # Stable temporal ordering within subjects
    df = df.sort_values([id_col, ycol, mcol], kind="mergesort").copy()

    # Temporal lags (within id_col)
    logging.info("Creating temporal lags for steps=%s…", lag_steps)

    # Only lag columns that actually exist
    base_cols_available = [c for c in WIDE_COLS if c in df.columns]
    missing_base = sorted(set(WIDE_COLS) - set(base_cols_available))
    if missing_base:
        logging.warning("Some base columns for lags are missing and will be skipped: %s", missing_base)

    if base_cols_available:
        for lag in lag_steps:
            lagged = df.groupby(id_col, observed=True)[base_cols_available].shift(lag)
            lagged.columns = [f"{col}_t-{lag}" for col in lagged.columns]
            df = pd.concat([df, lagged], axis=1)
    else:
        logging.warning("No base columns available for lagging. Skipping lag creation.")

    # -------- Valid date filtering & final cleanup
    if ycol in df.columns:
        df = df[df[ycol] >= min_year]

    # Remove any incomplete rows after lagging to ensure model-ready features
    df = df.dropna().reset_index(drop=True)

    # -------- Save
    logging.info("Saving Parquet to %s …", output_path)
    df.to_parquet(output_path, index=False)
    logging.info("Feature engineering completed with %d rows and %d columns.", len(df), df.shape[1])

    # Cleanup 
    del df
