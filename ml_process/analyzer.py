from libraries import permutation_importance, pd, shap, np, Path


def get_permutation_importance(models, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42):
    
    """
    Compute permutation importance for each model provided.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping {model_name: fitted_estimator}. Estimators must implement predict_proba
        or decision_function compatible with the chosen 'scoring'.
    X_test : pd.DataFrame
        Test features (columns used for importance ranking).
    y_test : pd.Series or array-like
        True labels for the test set.
    scoring : str, default 'roc_auc'
        Scikit-learn scoring metric passed to permutation_importance.
    n_repeats : int, default 10
        Number of random shuffles per feature.
    random_state : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by feature names; columns are model names.
        Values are normalized importances (each column divided by its max).
        Includes an extra column 'mean_importance' as the across-model average.
    """
    
    feature_importance = {}
    for name, model in models.items():
        # Compute permutation importance for one fitted model
        result = permutation_importance(model, X_test, y_test,n_repeats=n_repeats,random_state=random_state,scoring=scoring,n_jobs=-1)
        # Store mean importance across repeats
        feature_importance[name] = result.importances_mean
    # Assemble into a DataFrame aligned to X_test feature order
    importance_df = pd.DataFrame(feature_importance, index=X_test.columns)
    # Normalize per model (column-wise) to the [0, 1] range by max scaling
    importance_df = importance_df.div(importance_df.max())
    # Aggregate across models to get a consensus ranking
    importance_df['mean_importance'] = importance_df.mean(axis=1)
    return importance_df
    

def assemble_feature_summary(models, base_model_name, X_test, y_test, top_n=20):
    
    """
    Build a compact feature summary with:
    - Mean permutation importance across models.
    - Optional SHAP interaction-based "feature combo" for a base model.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping {model_name: fitted_estimator}. Estimators should be SHAP-compatible
        (for interactions) if you want the combo column populated.
    base_model_name : str
        Key in 'models' used to compute SHAP interaction values.
    X_test : pd.DataFrame
        Test features (used both for permutation importance and SHAP).
    y_test : pd.Series or array-like
        True labels (used for permutation importance).
    top_n : int, default 20
        Number of top features (by mean permutation importance) to include.

    Returns
    -------
    pd.DataFrame
        Columns:
          - 'Feature'        : feature name
          - 'Importance'     : mean permutation importance across models
          - 'Feature Combo'  : top 3 features interacting (by mean |interaction|) with 'Feature'
                               for the base model; 'n/a' if SHAP interactions are unavailable.
    """
    
    # Compute normalized permutation importance and pick the top_n features
    importance_df = get_permutation_importance(models, X_test, y_test)
    common_features = importance_df.nlargest(top_n, 'mean_importance').index.tolist()
    # Base model for SHAP interactions
    model = models[base_model_name]
    # 1) Importance vector for the selected features
    fi_series = importance_df.loc[common_features, 'mean_importance']
    # 2) Feature Combo via SHAP interaction values (tree-based models)
    try:
        # TreeExplainer is efficient for tree ensembles (XGB, LGBM, CatBoost, RF)
        explainer = shap.TreeExplainer(model)
        # SHAP interaction values: shape (n_samples, n_features, n_features)
        interaction_values = explainer.shap_interaction_values(X_test)
        # For multi-class, pick the class-1 cube (positive class) by convention
        if isinstance(interaction_values, list):  # multi-class output
            interaction_values = interaction_values[1]
        # Average absolute interaction across samples -> (n_features, n_features)
        interaction_matrix = np.abs(interaction_values).mean(axis=0)
        # For each top feature, get the 3 strongest partners (excluding itself)
        feature_combo_map = {}
        for feat in common_features:
            idx = X_test.columns.get_loc(feat)
            interactions = interaction_matrix[idx] # vector over all features
            sorted_idx = np.argsort(-interactions) # descending by interaction strength
            top_feats = [X_test.columns[j] for j in sorted_idx if X_test.columns[j] != feat][:3]
            feature_combo_map[feat] = ", ".join(top_feats)
        feature_combo_final = [feature_combo_map.get(f, "") for f in common_features]
    except Exception as e:
        # If SHAP interactions fail (e.g., model unsupported or memory constraints),
        # gracefully fall back to 'n/a' for the combo column.
        print(f"[SHAP Interaction] Warning: {e}")
        feature_combo_final = ["n/a"] * len(common_features)
    # Final summary table 
    feature_summary = pd.DataFrame({
        'Feature': common_features,
        'Importance': fi_series.values,
        'Feature Combo': feature_combo_final
    })
    return feature_summary
    

def save_feature_summary_txt(feature_summary: pd.DataFrame, file_path: Path):
    
    """
    Save the feature summary DataFrame into a human-readable TXT file.

    Parameters
    ----------
    feature_summary : pd.DataFrame
        DataFrame with columns ['Feature', 'Importance', 'Feature Combo'].
    file_path : str
        Destination path for the TXT file.
    """
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Feature Summary Report ===\n\n")
        f.write(f"Total features listed: {len(feature_summary)}\n\n")
        # Write table row by row
        f.write(f"{'Feature':<25}{'Importance':<15}{'Feature Combo'}\n")
        f.write("-" * 70 + "\n")
        for _, row in feature_summary.iterrows():
            f.write(f"{row['Feature']:<25}{row['Importance']:<15.4f}{row['Feature Combo']}\n")
    print(f"Feature summary saved to: {file_path}")




