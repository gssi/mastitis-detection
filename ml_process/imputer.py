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
- write_imputation_report: it gives in output the report containing the measures related to the difference between distributions pre- and post-imputation.
"""

from libraries import pd, Path, logging, wasserstein_distance, List, Dict, time, sns, plt, np
from collections import defaultdict


### STATIC VARIABLES ###

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



### FUNCTIONS ###

def apply_hierarchy(df: pd.DataFrame, col: str, hier: List[List[str]], log_map: Dict[str, List[str]]) -> pd.Series:
    
    """
    Impute a single column by progressively broader groupings (median per group).

    Parameters:
    
    - df : pd.DataFrame ---> input data.
    - col : str ---> column to impute.
    - hier : list[list[str]] ---> list of grouping strategies (from most specific to broadest).
    - log_map : dict[str, list[str]] ---> collector for logging which levels were used per variable.

    Returns:
    
    - pd.Series ---> copy of the column with NAs imputed where possible.

    Notes:
    
    - Groups with very few observations are skipped using a "per-group" support threshold (default: ≥3).
    - Only groups whose keys are fully present in df are attempted.
    """
    
    series = df[col].copy()
    if series.isna().sum() == 0:
        return series
    # minimal per-group support to trust the group median
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

    Returns int(IQS in {0, 1, 2, 3})
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
        corr_diff = (post_corr - pre_corr).abs().mean().mean()
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
    Wrapper to perform hierarchical imputation over CLINICAL_COLS using the provided hierarchies (IHG, PHG).
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
    # IQS evaluation only if we have enough cases
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


def write_imputation_report(input_path: Path, output_path: Path,*, features: List[str] = CLINICAL_COLS, wd_thresh: float = 0.10,
                            corr_thresh: float = 0.02, out_path: Path = Path("output/imputation_report.txt")) -> Path:

    """
    Generate a concise text report assessing whether imputation preserved
    the statistical structure of the data.

    The function compares the dataset BEFORE and AFTER imputation and
    summarizes both global and factor-wise stability using two metrics:
    - W/IQR  –> normalized Wasserstein distance (distributional similarity)
    - mean|Δρ| –> mean absolute change in Pearson correlations (structural stability)

    
    Output contents:
    
    1) Overall results per feature
       For each clinical variable:
         - %missing_pre  : missingness before imputation
         - W/IQR         : distributional shift (≤ 0.10 ideal)
         - mean|Δρ|      : correlation shift (≤ 0.02 ideal)

       Also reports the global mean|Δρ| across all features and lists
       any variables exceeding the thresholds.

    2) Factor-wise analyses
       Separate tables for each available factor:
         - lactation_phase
         - age
         - season
       Each table shows one row per level, with:
         - n_rows
         - mean_W/IQR  : average shift across features within that subgroup
         - mean|Δρ|    : average correlation change within the subgroup
         - % features exceeding each threshold

       These summaries highlight whether deviations are global or phase/age/season-specific.

    3) Summary verdict
       A short statement confirming if both global criteria
       (W/IQR ≤ 0.10 and mean|Δρ| ≤ 0.02) are satisfied.

    
    Returns path of location of the generated .txt report.
    """
    
    pre_df = pd.read_parquet(input_path)
    post_df = pd.read_parquet(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Helpers 
                                
    def iqr(x: np.ndarray) -> float:
        q1, q3 = np.nanpercentile(x, [25, 75])
        d = q3 - q1
        return float(d) if np.isfinite(d) and d > 0 else np.nan

    def wasserstein_norm(a: np.ndarray, b: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            return np.nan
        d = wasserstein_distance(a, b)
        i = iqr(a)  
        return float(d / i) if np.isfinite(i) and i > 0 else np.nan

    def pct_missing(s: pd.Series) -> float:
        return 100.0 * s.isna().mean()

    def pairwise_corr_delta_mean_abs(pre: pd.DataFrame, post: pd.DataFrame, cols: List[str]) -> tuple[float, pd.DataFrame]:
        if not cols:
            return np.nan, pd.DataFrame()
        pre_c = pre[cols].astype(float).corr(method="pearson", min_periods=10)
        post_c = post[cols].astype(float).corr(method="pearson", min_periods=10)
        if pre_c.notna().sum().sum() == 0 or post_c.notna().sum().sum() == 0:
            return np.nan, pd.DataFrame()
        delta = (post_c - pre_c).abs()
        tri = delta.where(np.triu(np.ones_like(delta, dtype=bool), 1))
        return float(np.nanmean(tri.values)), delta

    # Overall per-feature 
    rows = []
    for col in features:
        pre_vals = pre_df[col].to_numpy(dtype=float)
        post_vals = post_df[col].to_numpy(dtype=float)
        rows.append({
            "feature": col,
            "pct_missing_pre": round(pct_missing(pre_df[col]), 2),
            "pct_imputed": round(pct_missing(pre_df[col]), 2),
            "wasserstein_norm": round(wasserstein_norm(pre_vals, post_vals), 4)})
    overall_df = pd.DataFrame(rows)
    # Global correlation delta mean abs
    idx = post_df.index
    g_mean_delta, g_delta_mat = pairwise_corr_delta_mean_abs(pre_df.loc[idx], post_df.loc[idx], features)
    per_feat_mean_abs = {}
    if not g_delta_mat.empty:
        for f in features:
            others = [c for c in features if c != f]
            per_feat_mean_abs[f] = float(g_delta_mat.loc[f, others].mean())
    overall_df["pearson_corr_delta_mean_abs"] = overall_df["feature"].map(per_feat_mean_abs).round(4)
    viol_w = overall_df.loc[overall_df["wasserstein_norm"] > wd_thresh, "feature"].tolist()
    viol_rho = overall_df.loc[overall_df["pearson_corr_delta_mean_abs"] > corr_thresh, "feature"].tolist()
    # Factor-wise strata 
    factor_list = [("lactation_phase", "Phase"), ("age", "Age"), ("season", "Season")]
    factor_sections = []
    for key, title in factor_list:
        if key not in post_df.columns:
            continue
        levels = post_df[key].astype("object").astype(str)
        rows_sg = []
        for lvl, idx_sg in levels.groupby(levels).groups.items():
            idx_sg = pd.Index(idx_sg)
            n_rows = int(len(idx_sg))
            mean_delta_lvl, delta_mat_lvl = pairwise_corr_delta_mean_abs( pre_df.loc[idx_sg], post_df.loc[idx_sg], features)
            w_vals, rho_vals, viol_w_cnt, viol_rho_cnt = [], [], 0, 0
            for col in features:
                w = wasserstein_norm(
                    pre_df.loc[idx_sg, col].to_numpy(dtype=float),
                    post_df.loc[idx_sg, col].to_numpy(dtype=float))
                if np.isfinite(w):
                    w_vals.append(w)
                    if w > wd_thresh:
                        viol_w_cnt += 1
                per_feat = np.nan
                if not delta_mat_lvl.empty and (col in delta_mat_lvl.index):
                    others = [c for c in features if c != col and c in delta_mat_lvl.columns]
                    if others:
                        per_feat = float(delta_mat_lvl.loc[col, others].mean())
                        if np.isfinite(per_feat):
                            rho_vals.append(per_feat)
                            if per_feat > corr_thresh:
                                viol_rho_cnt += 1
            mean_w = float(np.nanmean(w_vals)) if w_vals else np.nan
            mean_rho = float(np.nanmean(rho_vals)) if rho_vals else np.nan
            share_viol_w = (viol_w_cnt / len(features)) * 100 if len(features) else np.nan
            share_viol_rho = (viol_rho_cnt / len(features)) * 100 if len(features) else np.nan
            rows_sg.append({
                "level": str(lvl),
                "n_rows": n_rows,
                "mean_WIQR": round(mean_w, 4) if np.isfinite(mean_w) else np.nan,
                "mean_meanAbsDeltaR": round(mean_rho, 4) if np.isfinite(mean_rho) else np.nan,
                "pct_feat_viol_WIQR": round(share_viol_w, 1) if np.isfinite(share_viol_w) else np.nan,
                "pct_feat_viol_meanAbsDeltaR": round(share_viol_rho, 1) if np.isfinite(share_viol_rho) else np.nan})
        sg_df = pd.DataFrame(rows_sg)
        if not sg_df.empty:
            sg_df = sg_df.sort_values(["pct_feat_viol_WIQR", "pct_feat_viol_meanAbsDeltaR", "mean_WIQR"], ascending=[False, False, False])
        factor_sections.append((title, key, sg_df))
    # Report (.txt)
    lines = []
    lines.append("IMPUTATION REPORT\n")
    lines.append("\n")
    lines.append("Metrics:\n")
    lines.append(f"- Distributional similarity: Normalized Wasserstein distance (W/IQR) ≤ {wd_thresh}\n")
    lines.append(f"- Multivariate preservation: mean absolute difference between correlation matrices (Pearson) ≤ {corr_thresh}\n\n")
    # Overall
    lines.append("1) Overall results per feature\n")
    header = f"{'feature':<20}{'%missing_pre':>14}{'%imputed':>12}{'W/IQR':>12}{'mean|Δρ|':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in overall_df.iterrows():
        lines.append(
            f"{str(r['feature']):<20}"
            f"{r['pct_missing_pre']:>14.2f}"
            f"{r['pct_imputed']:>12.2f}"
            f"{(r['wasserstein_norm'] if pd.notna(r['wasserstein_norm']) else np.nan):>12.4f}"
            f"{(r['pearson_corr_delta_mean_abs'] if pd.notna(r['pearson_corr_delta_mean_abs']) else np.nan):>12.4f}")
    lines.append("\n")
    if np.isfinite(g_mean_delta):
        lines.append(f"Global mean |Δρ| (Pearson) over {len(features)} features: {g_mean_delta:.4f}\n")
    else:
        lines.append("Global mean |Δρ| (Pearson): not available (insufficient pairwise information).\n")
    lines.append(f"Features exceeding W/IQR threshold: {viol_w if viol_w else 'none'}\n")
    lines.append(f"Features exceeding mean |Δρ| threshold: {viol_rho if viol_rho else 'none'}\n")
    # Factor-wise
    lines.append("\n2) Factor-wise subgroup analysis (separate)\n")
    for title, key, sg_df in factor_sections:
        lines.append(f"\n– Subgroups by {title} [{key}]\n")
        if sg_df is None or sg_df.empty:
            lines.append("No eligible subgroups.\n")
            continue
        head = f"{'level':<24}{'n_rows':>10}{'mean_W/IQR':>14}{'mean|Δρ|':>12}{'%feat_viol_W':>14}{'%feat_viol_Δρ':>16}"
        lines.append(head)
        lines.append("-" * len(head))
        for _, r in sg_df.iterrows():
            lines.append(
                f"{str(r['level']):<24}"
                f"{int(r['n_rows']):>10}"
                f"{(r['mean_WIQR'] if pd.notna(r['mean_WIQR']) else np.nan):>14.4f}"
                f"{(r['mean_meanAbsDeltaR'] if pd.notna(r['mean_meanAbsDeltaR']) else np.nan):>12.4f}"
                f"{(r['pct_feat_viol_WIQR'] if pd.notna(r['pct_feat_viol_WIQR']) else np.nan):>14}"
                f"{(r['pct_feat_viol_meanAbsDeltaR'] if pd.notna(r['pct_feat_viol_meanAbsDeltaR']) else np.nan):>16}")
        lines.append("")
    ok_w = all(pd.isna(v) or v <= wd_thresh for v in overall_df["wasserstein_norm"])
    ok_r = (np.isfinite(g_mean_delta) and g_mean_delta <= corr_thresh) and \
           all(pd.isna(v) or v <= corr_thresh for v in overall_df["pearson_corr_delta_mean_abs"])
    lines.append("\n3) Summary (global) verdict\n")
    if ok_w and ok_r:
        lines.append("Post-imputation distributions are consistent (W/IQR within threshold) and the correlation structure is preserved (mean |Δρ| within threshold).")
    else:
        lines.append("Some indicators exceed acceptance thresholds. Inspect factor-wise sections above.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Imputation report written to: %s", str(out_path))
    return out_path











