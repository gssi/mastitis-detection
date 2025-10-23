"""
Hierarchical Clinical Imputation

This module performs hierarchical imputation of missing clinical variables using
predefined grouping strategies and group-wise medians. The imputation proceeds
through multiple aggregation levels (from specific to general), with sensible
fallbacks and quality checks.

Main functionalities:
- apply_hierarchy: fill missing values using progressively broader groupings.
- hierarchical_block: multi-variable imputation with IHG → PHG → IHG fallbacks.
- _iqs: imputation quality score (distributional shift, correlation preservation, range validity).
- clinical_impute_df: wrapper to impute clinical columns using predefined hierarchies.
- run_clinical_imputation: full workflow from IO to IQS evaluation and saving.
- distribution_comparison: KDE plots to visually compare pre/post distributions.
"""

from libraries import pd, Path, logging, wasserstein_distance, List, Dict, time, sns, plt, np
from collections import defaultdict

# =========================
# STATIC VARIABLES
# =========================

# Clinical variables to impute
CLINICAL_COLS = ["scs", "milk", "protein", "fat", "lactose", "ec"]

# Plausible ranges used for post-imputation sanity checks
RANGES = {
    "scs": (0, 9),
    "milk": (0, 700),
    "protein": (0, 6),
    "fat": (0, 8),
    "lactose": (4, 5.5),
    "ec": (500, 2000),
}

# Individual-based hierarchy (IHG): starts highly specific and backs off
IHG = [
    ["id", "mastitis", "lactation_phase", "age", "season"],
    ["id", "mastitis", "lactation_phase", "age"],
    ["id", "mastitis", "lactation_phase"],
    ["id", "mastitis"],
]

# Population-based hierarchy (PHG): broader, includes breed/healthy status
PHG = [
    ["mastitis", "healthy", "lactation_phase", "age", "season", "breed"],
    ["mastitis", "healthy", "lactation_phase", "age", "season"],
    ["mastitis", "healthy", "lactation_phase", "age"],
    ["mastitis", "healthy", "lactation_phase"],
]


# =========================
# FUNCTIONS
# =========================

def apply_hierarchy(df: pd.DataFrame, col: str, hier: List[List[str]], log_map: Dict[str, List[str]]) -> pd.Series:
    """
    Impute a single column by progressively broader groupings (median per group).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col : str
        Column to impute.
    hier : list[list[str]]
        List of grouping strategies (from most specific to broadest).
    log_map : dict[str, list[str]]
        Collector for logging which levels were used per variable.

    Returns
    -------
    pd.Series
        A copy of the column with NAs imputed where possible.

    Notes
    -----
    - Groups with very few observations are skipped using a **per-group** support threshold (default: ≥3).
    - Only groups whose keys are fully present in df are attempted.
    """
    series = df[col].copy()
    if series.isna().sum() == 0:
        return series

    # minimal per-group support to trust the group's median
    min_group_n = 3  # tune to 2/4/5 

    for grp in hier:
        if series.isna().sum() == 0:
            break
        if not set(grp).issubset(df.columns):
            continue

        # dropna=True to avoid NaN-key groups
        g = df.groupby(grp, dropna=True)[col]

        # per-group threshold instead of "median(count) at level"
        counts = g.transform("count")
        medians = g.transform("median")

        # fill only where: value is NaN, a group median exists, and the group's count >= threshold
        newly_filled = series.isna() & medians.notna() & (counts >= min_group_n)

        if newly_filled.any():
            # optional: include threshold info in the log
            log_map[col].append(" → ".join(grp) + f" [per-group n≥{min_group_n}]")
            series.loc[newly_filled] = medians.loc[newly_filled]

        # continue to next (broader) level for any remaining NaNs
    return series


def hierarchical_block(df: pd.DataFrame, cols: List[str], hier1: List[List[str]], hier2: List[List[str]]) -> pd.DataFrame:
    """
    Apply hierarchical imputation to multiple columns with sensible fallbacks.

    Strategy per column:
      IHG (specific) → PHG (broader) → IHG (final attempt)
    """
    imput_log = defaultdict(list)

    for c in cols:
        if df[c].isna().sum() == 0:
            continue
        # Try IHG → PHG → fallback IHG
        df[c] = apply_hierarchy(df, c, hier1, imput_log)
        if df[c].isna().sum() > 0:
            df[c] = apply_hierarchy(df, c, hier2, imput_log)
        if df[c].isna().sum() > 0:
            df[c] = apply_hierarchy(df, c, hier1, imput_log)
        if df[c].isna().sum() > 0:
            logging.warning("Variable '%s' still has %d missing values post-imputation.", c, int(df[c].isna().sum()))

    # Final report: which levels were actually used for each variable
    for var, levels in imput_log.items():
        logging.info("'%s' imputed using %d levels: %s", var, len(levels), levels)

    return df


def _iqs(pre: pd.DataFrame, post: pd.DataFrame, cols: List[str]) -> int:
    """
    Compute a 3-point Imputation Quality Score (IQS):
      1) Distributional shift (normalized Wasserstein distance) <= 0.1 for all cols.
      2) Mean absolute correlation difference <= 0.02.
      3) No values outside plausible ranges.

    Returns
    -------
    int
        IQS in {0, 1, 2, 3}
    """

    score = 0
    WD_THRESH = 0.10

    # 1) Normalized Wasserstein distance (robust to scale via IQR normalization)
    wd = {}
    for c in cols:
        x = pre[c]
        y = post[c]
        # Use only rows where both are non-NaN (local NaN handling)
        m = x.notna() & y.notna()
        if m.sum() == 0:
            wd[c] = float("nan")
            continue
        x = x[m]
        y = y[m]
        iqr = (x.quantile(0.75) - x.quantile(0.25)) + 1e-8 # 1e-8 to prevent division by zero
        dist = wasserstein_distance(x.values, y.values) / iqr
        wd[c] = float(dist)

    bad_cols = [c for c, d in wd.items() if (np.isfinite(d) and d > WD_THRESH)]
    if not bad_cols:
        score += 1
    else:
        logging.warning("IQS criterion #1 violated: %d vars with normalized distance > %.2f",len(bad_cols), WD_THRESH)
        for c in bad_cols:
            logging.info("  - '%s': normalized distance = %.4f", c, wd[c])

    # 2) Correlation preservation 
    common = pre[cols].notna().all(axis=1) & post[cols].notna().all(axis=1)
    if common.sum() > 2:  # servono almeno 3 righe per corr
        pre_corr = pre.loc[common, cols].corr()
        post_corr = post.loc[common, cols].corr()
        corr_diff = (pre_corr - post_corr).abs().mean().mean()
    else:
        corr_diff = np.inf
        logging.warning("IQS criterion #2 skipped: insufficient common complete rows (n=%d).",int(common.sum()))

    if corr_diff <= 0.02:
        score += 1
    else:
        logging.warning("IQS criterion #2 violated: mean |Δcorr| = %.4f", float(corr_diff))
        if np.isfinite(corr_diff):
            corr_matrix_diff = (pre_corr - post_corr).abs()
            for c in cols:
                mean_diff = float(corr_matrix_diff[c].mean())
                if mean_diff > 0.02:
                    logging.info("  - '%s': mean |Δcorr| = %.4f", c, mean_diff)

    # 3) Range validity (post-imputation plausibility)
    out_of_range = False
    for c in cols:
        if c in RANGES:
            lo, hi = RANGES[c]
            violations = int(((post[c] < lo) | (post[c] > hi)).sum())
            if violations > 0:
                logging.warning("IQS criterion #3 violated: '%s' has %d out-of-range values [%s, %s]",c, violations, lo, hi)
                out_of_range = True
    if not out_of_range:
        score += 1

    return int(score)


def clinical_impute_df(df: pd.DataFrame, hier1=IHG, hier2=PHG) -> pd.DataFrame:
    """
    Wrapper to perform hierarchical imputation over CLINICAL_COLS
    using the provided hierarchies (IHG, PHG).
    """
    logging.info("Starting hierarchical imputation…")
    t0 = time.time()
    df = hierarchical_block(df, CLINICAL_COLS, hier1, hier2)
    dt = time.time() - t0
    logging.info("Imputation completed in %.2fs", dt)
    return df


def run_clinical_imputation(input_path: Path, output_path: Path, sample_frac_iqs: float = 1.0) -> None:
    """
    Orchestrate the full imputation workflow:
      - Load dataset
      - Build a baseline for IQS evaluation (optional sampling)
      - Impute
      - Evaluate IQS (if sample large enough)
      - Save results
    """
    df = pd.read_parquet(input_path)

    # Build base for IQS evaluation
    base = df[CLINICAL_COLS]
    if 0 < sample_frac_iqs <= 1.0 and base.shape[0] > 0:
        base = base.sample(frac=sample_frac_iqs, random_state=42)

    logging.info("Starting imputation…")
    df_imp = clinical_impute_df(df)

    # IQS evaluation only if we have enough complete cases
    if base.shape[0] > 100:
        logging.info("Evaluating IQS…")
        iqs = _iqs(base, df_imp.loc[base.index, CLINICAL_COLS], CLINICAL_COLS)
        logging.info("IQS = %d/3", iqs)
    else:
        logging.info("IQS skipped (insufficient complete cases: %d)", base.shape[0])

    logging.info("Saving imputed dataset…")
    df_imp.to_parquet(output_path, index=False)
    logging.info("Imputation pipeline completed.")

    # Cleanup
    del df, df_imp, base


def distribution_comparison(input_path: Path, output_path: Path, max_points: int = 20000) -> None:
    """
    Visual comparison of distributions (Pre vs Post) via KDE plots.
    Uses a safe sample to avoid overplotting on very large datasets.
    """

    def safe_sample(series: pd.Series) -> pd.Series:
        s = series.dropna()
        return s.sample(n=min(len(s), max_points), random_state=42) if len(s) else s

    df_pre = pd.read_parquet(input_path)
    df_post = pd.read_parquet(output_path)

    variables = CLINICAL_COLS
    n_cols = 2
    n_rows = (len(variables) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        pre_data = safe_sample(df_pre[var])
        post_data = safe_sample(df_post[var])

        # KDE overlays (Pre vs Post)
        sns.kdeplot(pre_data, label="Pre", color="red", fill=True, alpha=0.5, ax=ax, bw_adjust=1)
        sns.kdeplot(post_data, label="Post", color="blue", fill=True, alpha=0.5, ax=ax, bw_adjust=1)

        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.legend()

    # Remove unused subplots if variables do not fill the grid
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Cleanup
    del df_pre, df_post, variables, fig, axes





