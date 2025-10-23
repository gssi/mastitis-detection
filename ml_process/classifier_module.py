"""
Model training utilities for clinical mastitis prediction.

This module provides:
- A group-aware train/test split by animal ID (no leakage across sets).
- A simple repeated group K-fold wrapper for CV with grouped samples.
- Training/evaluation routines for XGBoost, Random Forest, CatBoost, and LightGBM
  with grid search, group-aware CV, and standardized reporting.
- Utilities to load previously saved classifiers from disk.

Notes:
- Grouping by animal ID is critical to avoid optimistic bias.
- Metrics are reported both via CV and on a held-out test set.
"""

from libraries import (
    pd, train_test_split, gc, multiprocessing, xgb, np,
    GridSearchCV, accuracy_score, precision_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix, RandomForestClassifier, CatBoostClassifier,
    lgb, os, pickle, make_scorer, cross_validate
)
from pathlib import Path


# =========================
# DATA SPLIT
# =========================

def split_by_animal(input_path: Path, target_col: str = 'mastitis',
                    test_size: float = 0.2, random_state: int = 42):
                      
    """
    Grouped (by animal ID) train/test split to prevent leakage.

    Parameters
    ----------
    input_path : Path
        Parquet path with features + target; must contain 'id' and `target_col`.
    target_col : str, default 'mastitis'
        Name of the binary target column.
    test_size : float, default 0.2
        Fraction of animals placed in the test set.
    random_state : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    X_train, y_train, X_test, y_test : pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        Feature matrices and targets for train and test sets.

    Notes
    -----
    - Excludes obvious leakage columns (lags, date parts, id, healthy flags).
    """
                      
    df = pd.read_parquet(input_path).copy()
    # Columns to exclude from the models (potential leakage / identifiers)
    exclude_cols = [
        'id', target_col, f'{target_col}_t-1', f'{target_col}_t-2',
        'healthy', 'month', 'year', 'month_t-1', 'month_t-2', 'year_t-1',
        'year_t-2', 'age_cat'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Split based on unique animal IDs
    animal_ids = df['id'].unique()
    train_ids, test_ids = train_test_split(
        animal_ids, test_size=test_size, random_state=random_state
    )
    # Row selection by IDs
    train_df = df[df['id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['id'].isin(test_ids)].reset_index(drop=True)
    # X and y
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)
    # Report
    print("Split completed:")
    print(f"Train set: {len(X_train)} rows - {train_df['id'].nunique()} animals")
    print(f"Test set:  {len(X_test)} rows - {test_df['id'].nunique()} animals")
    print(f"Positive class (train): {y_train.sum()} / {len(y_train)} ({100 * y_train.mean():.2f}%)")
    print(f"Positive class (test):  {y_test.sum()} / {len(y_test)} ({100 * y_test.mean():.2f}%)")
    # Cleanup
    del train_df, test_df, train_ids, animal_ids, feature_cols, df
    gc.collect()
    return X_train, y_train, X_test, y_test


# =========================
# GROUPED CV WRAPPER
# =========================

class RepeatedGroupKFoldWrapper:
  
    """
    Simple repeated K-fold over groups (animal IDs).

    Shuffle the unique groups at each repeat; split into `n_splits` folds; yield
    indices corresponding to train/test groups.

    Parameters
    ----------
    groups : array-like
        Group labels for each sample (length must match X).
    n_splits : int, default 5
        Number of folds.
    n_repeats : int, default 3
        Number of repetitions.
    random_state : int, default 42
        RNG seed for reproducibility.
    """
  
    def __init__(self, groups, n_splits: int = 5, n_repeats: int = 3, random_state: int = 42):
        self.groups = np.array(groups)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        unique_groups = np.unique(self.groups)
        for _ in range(self.n_repeats):
            rng.shuffle(unique_groups)
            folds = np.array_split(unique_groups, self.n_splits)
            for i in range(self.n_splits):
                test_groups = folds[i]
                train_groups = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
                train_idx = np.where(np.isin(self.groups, train_groups))[0]
                test_idx = np.where(np.isin(self.groups, test_groups))[0]
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


# =========================
# MODELS: XGB / RF / CAT / LGBM
# =========================

def call_xgb(X_train_scaled: pd.DataFrame, y_train: pd.Series,
             X_test_scaled: pd.DataFrame, y_test: pd.Series, groups):
               
    """
    Train/evaluate XGBoost with group-aware CV and a (fixed) param grid.
    Saves the best estimator to `classifier/xgb_model.pkl` and a plain-text
    metrics report to `output/xgb_report.txt`.
    """
               
    # 1) Available cores
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    # 2) Base model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=num_cores
    )
    # 3) Param grid (fixed with best values after Grid Search)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [8],
        'learning_rate': [0.05],
        'subsample': [0.7],
        'colsample_bytree': [0.8],
        'gamma': [5],
        'reg_alpha': [2],
        'reg_lambda': [5]
    }
    # 4) Grouped CV
    cv = RepeatedGroupKFoldWrapper(groups=groups, n_splits=5, n_repeats=3, random_state=42)
    # 5) Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=num_cores
    )
    # 6) Fit
    grid_search.fit(X_train_scaled, y_train)
    best_model_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    # 7) Cross-validation metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    }
    cv_scores = cross_validate(
        best_model_xgb, X_train_scaled, y_train,
        groups=groups, cv=cv, scoring=scoring,
        n_jobs=num_cores, return_train_score=False
    )
    print("\n Cross-Validation (mean ± std):")
    cv_summary = {}
    for metric in scoring.keys():
        scores = cv_scores[f'test_{metric}']
        m, s = scores.mean(), scores.std()
        cv_summary[metric] = (m, s)
        print(f"{metric.capitalize():<10}: {m:.4f} ± {s:.4f}")
    # 8) Test set
    y_pred = best_model_xgb.predict(X_test_scaled)
    y_pred_proba = best_model_xgb.predict_proba(X_test_scaled)[:, 1]
    acc_test  = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test  = recall_score(y_test, y_pred, zero_division=0)
    f1_test   = f1_score(y_test, y_pred, zero_division=0)
    auc_test  = roc_auc_score(y_test, y_pred_proba)
    cm_test   = confusion_matrix(y_test, y_pred)

    print("\n Test Set Metrics (XGB):")
    print(f"Accuracy:  {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall:    {rec_test:.4f}")
    print(f"F1 Score:  {f1_test:.4f}")
    print(f"ROC AUC:   {auc_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)

    # 9) Train set (sanity check)
    y_train_pred = best_model_xgb.predict(X_train_scaled)
    y_train_pred_proba = best_model_xgb.predict_proba(X_train_scaled)[:, 1]
    acc_tr  = accuracy_score(y_train, y_train_pred)
    prec_tr = precision_score(y_train, y_train_pred, zero_division=0)
    rec_tr  = recall_score(y_train, y_train_pred, zero_division=0)
    f1_tr   = f1_score(y_train, y_train_pred, zero_division=0)
    auc_tr  = roc_auc_score(y_train, y_train_pred_proba)
    cm_tr   = confusion_matrix(y_train, y_train_pred)

    print("\n Training Set Metrics (XGB):")
    print(f"Accuracy:  {acc_tr:.4f}")
    print(f"Precision: {prec_tr:.4f}")
    print(f"Recall:    {rec_tr:.4f}")
    print(f"F1 Score:  {f1_tr:.4f}")
    print(f"ROC AUC:   {auc_tr:.4f}")
    print("Confusion Matrix:")
    print(cm_tr)

    # 10) Save model
    os.makedirs('classifier', exist_ok=True)
    model_path = os.path.join('classifier', 'xgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_xgb, f)
    print(f"\n Model saved to: {model_path}")
    # 11) Save a clean text report with metrics
    report_path = os.path.join('output', 'xgb_report.txt')
    lines = []
    lines.append("XGBoost – Training Report\n")
    lines.append("Best Hyperparameters (Grid Search):")
    for k, v in best_params.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Cross-Validation (mean ± std)")
    for metric, (m, s) in cv_summary.items():
        lines.append(f"  {metric:>9}: {m:.4f} ± {s:.4f}")
    lines.append("")
    lines.append("Test Set Metrics")
    lines.append(f"  Accuracy : {acc_test:.4f}")
    lines.append(f"  Precision: {prec_test:.4f}")
    lines.append(f"  Recall   : {rec_test:.4f}")
    lines.append(f"  F1 Score : {f1_test:.4f}")
    lines.append(f"  ROC AUC  : {auc_test:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_test[0,0]}  {cm_test[0,1]}]")
    lines.append(f"     [{cm_test[1,0]}  {cm_test[1,1]}]]")
    lines.append("")
    lines.append("Training Set Metrics")
    lines.append(f"  Accuracy : {acc_tr:.4f}")
    lines.append(f"  Precision: {prec_tr:.4f}")
    lines.append(f"  Recall   : {rec_tr:.4f}")
    lines.append(f"  F1 Score : {f1_tr:.4f}")
    lines.append(f"  ROC AUC  : {auc_tr:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_tr[0,0]}  {cm_tr[0,1]}]")
    lines.append(f"     [{cm_tr[1,0]}  {cm_tr[1,1]}]]")
    lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f" Metrics report saved to: {report_path}")
               

def call_rf(X_train_scaled: pd.DataFrame, y_train: pd.Series,
            X_test_scaled: pd.DataFrame, y_test: pd.Series, groups):
              
    """
    Train/evaluate Random Forest with group-aware CV and a (fixed) param grid.
    Saves the best estimator to `classifier/rf_model.pkl` and a plain-text
    metrics report to `output/rf_report.txt`.
    """
              
    # 1) Param grid (fixed with best values after Grid Search)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [8],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'max_features': ['sqrt'],
        'bootstrap': [True],
    }
    # 2) Base model
    rf_base = RandomForestClassifier(
        random_state=42,
        verbose=0,
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
    )
    # 3) Grouped CV
    cv = RepeatedGroupKFoldWrapper(groups=groups, n_splits=5, n_repeats=3, random_state=42)
    # 4) Grid search
    print(" Starting RF Grid Search...")
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)
    # 5) Best params/model
    best_params = grid_search.best_params_
    print(f"\n Best hyperparameters (RF):\n{best_params}")
    best_model_rf = RandomForestClassifier(
        **best_params,
        random_state=42,
        verbose=0,
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
    )
    # 6) Fit on full train
    best_model_rf.fit(X_train_scaled, y_train)
    # 7) CV metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    }
    cv_scores = cross_validate(
        best_model_rf, X_train_scaled, y_train,
        groups=groups, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    print("\n Cross-Validation (mean ± std):")
    cv_summary = {}
    for metric in scoring.keys():
        scores = cv_scores[f'test_{metric}']
        m, s = scores.mean(), scores.std()
        cv_summary[metric] = (m, s)
        print(f"{metric.capitalize():<10}: {m:.4f} ± {s:.4f}")
    # 8) Test set
    y_pred = best_model_rf.predict(X_test_scaled)
    y_pred_proba = best_model_rf.predict_proba(X_test_scaled)[:, 1]
    acc_test  = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test  = recall_score(y_test, y_pred, zero_division=0)
    f1_test   = f1_score(y_test, y_pred, zero_division=0)
    auc_test  = roc_auc_score(y_test, y_pred_proba)
    cm_test   = confusion_matrix(y_test, y_pred)
              
    print("\n Test Set Metrics (RF):")
    print(f"Accuracy:  {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall:    {rec_test:.4f}")
    print(f"F1 Score:  {f1_test:.4f}")
    print(f"ROC AUC:   {auc_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)

    # 9) Train set
    y_train_pred = best_model_rf.predict(X_train_scaled)
    y_train_pred_proba = best_model_rf.predict_proba(X_train_scaled)[:, 1]
    acc_tr  = accuracy_score(y_train, y_train_pred)
    prec_tr = precision_score(y_train, y_train_pred, zero_division=0)
    rec_tr  = recall_score(y_train, y_train_pred, zero_division=0)
    f1_tr   = f1_score(y_train, y_train_pred, zero_division=0)
    auc_tr  = roc_auc_score(y_train, y_train_pred_proba)
    cm_tr   = confusion_matrix(y_train, y_train_pred)

    print("\n Training Set Metrics (RF):")
    print(f"Accuracy:  {acc_tr:.4f}")
    print(f"Precision: {prec_tr:.4f}")
    print(f"Recall:    {rec_tr:.4f}")
    print(f"F1 Score:  {f1_tr:.4f}")
    print(f"ROC AUC:   {auc_tr:.4f}")
    print("Confusion Matrix:")
    print(cm_tr)

    # 10) Save model
    os.makedirs('classifier', exist_ok=True)
    model_path = os.path.join('classifier', 'rf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_rf, f)
    print(f"\n Model saved to: {model_path}")
    # 11) Save a clean text report with metrics
    report_path = os.path.join('output', 'rf_report.txt')
    lines = []
    lines.append("Random Forest – Training Report\n")
    lines.append("Best Hyperparameters (Grid Search):")
    for k, v in best_params.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Cross-Validation (mean ± std)")
    for metric, (m, s) in cv_summary.items():
        lines.append(f"  {metric:>9}: {m:.4f} ± {s:.4f}")
    lines.append("")
    lines.append("Test Set Metrics")
    lines.append(f"  Accuracy : {acc_test:.4f}")
    lines.append(f"  Precision: {prec_test:.4f}")
    lines.append(f"  Recall   : {rec_test:.4f}")
    lines.append(f"  F1 Score : {f1_test:.4f}")
    lines.append(f"  ROC AUC  : {auc_test:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_test[0,0]}  {cm_test[0,1]}]")
    lines.append(f"     [{cm_test[1,0]}  {cm_test[1,1]}]]")
    lines.append("")
    lines.append("Training Set Metrics")
    lines.append(f"  Accuracy : {acc_tr:.4f}")
    lines.append(f"  Precision: {prec_tr:.4f}")
    lines.append(f"  Recall   : {rec_tr:.4f}")
    lines.append(f"  F1 Score : {f1_tr:.4f}")
    lines.append(f"  ROC AUC  : {auc_tr:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_tr[0,0]}  {cm_tr[0,1]}]")
    lines.append(f"     [{cm_tr[1,0]}  {cm_tr[1,1]}]]")
    lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f" Metrics report saved to: {report_path}")
              

def call_cat(X_train_scaled: pd.DataFrame, y_train: pd.Series,
             X_test_scaled: pd.DataFrame, y_test: pd.Series, groups):
               
    """
    Train/evaluate CatBoost with group-aware CV and a (fixed) param grid.
    Saves:
      - best estimator to  `classifier/cat_model.pkl`
      - plain-text metrics to `output/cat_report.txt`
    """
               
    # 1) Param grid (fixed with best values after Grid Search)
    param_grid = {
        'iterations': [300],
        'learning_rate': [0.05],
        'depth': [8],
        'l2_leaf_reg': [5],
        'border_count': [64],
        'bagging_temperature': [1],
        'random_strength': [5],
    }
    # 2) Base model
    cat_model = CatBoostClassifier(
        loss_function='Logloss',
        boosting_type='Ordered',
        eval_metric='AUC',
        allow_writing_files=False,
        verbose=200,
        random_seed=42,
        thread_count=max(1, multiprocessing.cpu_count() - 2),
    )
    # 3) Grouped CV
    cv = RepeatedGroupKFoldWrapper(groups=groups, n_splits=5, n_repeats=3, random_state=42)
    # 4) Grid search
    print(" Starting CatBoost Grid Search...")
    grid_search = GridSearchCV(
        cat_model, param_grid, cv=cv, scoring='f1', verbose=2, n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    # 5) Best params & model
    best_params = grid_search.best_params_
    print(f"\n Best hyperparameters (CatBoost):\n{best_params}")
    best_model_cat = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="AUC",
        allow_writing_files=False,
        random_seed=42,
        thread_count=max(1, multiprocessing.cpu_count() - 2),
        verbose=100,
    )
    best_model_cat.fit(X_train_scaled, y_train)
    # 6) CV metrics (mean ± std)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    }
    cv_scores = cross_validate(
        best_model_cat, X_train_scaled, y_train,
        groups=groups, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    print("\n Cross-Validation (mean ± std):")
    cv_summary = {}
    for metric in scoring.keys():
        scores = cv_scores[f'test_{metric}']
        m, s = scores.mean(), scores.std()
        cv_summary[metric] = (m, s)
        print(f"{metric.capitalize():<10}: {m:.4f} ± {s:.4f}")
    # 7) Test set metrics
    y_pred = best_model_cat.predict(X_test_scaled)
    y_pred_proba = best_model_cat.predict_proba(X_test_scaled)[:, 1]
    acc_test  = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test  = recall_score(y_test, y_pred, zero_division=0)
    f1_test   = f1_score(y_test, y_pred, zero_division=0)
    auc_test  = roc_auc_score(y_test, y_pred_proba)
    cm_test   = confusion_matrix(y_test, y_pred)

    print("\n Test Set Metrics (CatBoost):")
    print(f"Accuracy:  {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall:    {rec_test:.4f}")
    print(f"F1 Score:  {f1_test:.4f}")
    print(f"ROC AUC:   {auc_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)

    # 8) Train set metrics
    y_train_pred = best_model_cat.predict(X_train_scaled)
    y_train_pred_proba = best_model_cat.predict_proba(X_train_scaled)[:, 1]
    acc_tr  = accuracy_score(y_train, y_train_pred)
    prec_tr = precision_score(y_train, y_train_pred, zero_division=0)
    rec_tr  = recall_score(y_train, y_train_pred, zero_division=0)
    f1_tr   = f1_score(y_train, y_train_pred, zero_division=0)
    auc_tr  = roc_auc_score(y_train, y_train_pred_proba)
    cm_tr   = confusion_matrix(y_train, y_train_pred)

    print("\n Training Set Metrics (CatBoost):")
    print(f"Accuracy:  {acc_tr:.4f}")
    print(f"Precision: {prec_tr:.4f}")
    print(f"Recall:    {rec_tr:.4f}")
    print(f"F1 Score:  {f1_tr:.4f}")
    print(f"ROC AUC:   {auc_tr:.4f}")
    print("Confusion Matrix:")
    print(cm_tr)

    # 9) Save model
    os.makedirs('classifier', exist_ok=True)
    model_path = os.path.join('classifier', 'cat_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_cat, f)
    print(f"\n Model saved to: {model_path}")
    # 10) Save plain-text report
    report_path = os.path.join('output', 'cat_report.txt')
    lines = []
    lines.append("CatBoost – Training Report\n")
    lines.append("Best Hyperparameters (Grid Search):")
    for k, v in best_params.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Cross-Validation (mean ± std)")
    for metric, (m, s) in cv_summary.items():
        lines.append(f"  {metric:>9}: {m:.4f} ± {s:.4f}")
    lines.append("")
    lines.append("Test Set Metrics")
    lines.append(f"  Accuracy : {acc_test:.4f}")
    lines.append(f"  Precision: {prec_test:.4f}")
    lines.append(f"  Recall   : {rec_test:.4f}")
    lines.append(f"  F1 Score : {f1_test:.4f}")
    lines.append(f"  ROC AUC  : {auc_test:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_test[0,0]}  {cm_test[0,1]}]")
    lines.append(f"     [{cm_test[1,0]}  {cm_test[1,1]}]]")
    lines.append("")
    lines.append("Training Set Metrics")
    lines.append(f"  Accuracy : {acc_tr:.4f}")
    lines.append(f"  Precision: {prec_tr:.4f}")
    lines.append(f"  Recall   : {rec_tr:.4f}")
    lines.append(f"  F1 Score : {f1_tr:.4f}")
    lines.append(f"  ROC AUC  : {auc_tr:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_tr[0,0]}  {cm_tr[0,1]}]")
    lines.append(f"     [{cm_tr[1,0]}  {cm_tr[1,1]}]]")
    lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Metrics report saved to: {report_path}")
               

def call_lgbm(X_train_scaled: pd.DataFrame, y_train: pd.Series,
              X_test_scaled: pd.DataFrame, y_test: pd.Series, groups):
                
    """
    Train/evaluate LightGBM with group-aware CV and a (fixed) param grid.
    Saves:
      - best estimator to  `classifier/lgbm_model.pkl`
      - plain-text metrics to `output/lgbm_report.txt`
    """
                
    # 1) Param grid (fixed with best values after Grid Search)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [8],
        'learning_rate': [0.05],
        'feature_fraction': [0.7],
        'bagging_fraction': [0.7],
        'bagging_freq': [1],
        'min_data_in_leaf': [50],
        'max_bin': [255],
        'lambda_l1': [0.3],
        'lambda_l2': [1.0],
    }
    # 2) Base model
    lgbm_model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        is_unbalance=False,
        random_state=42,
        metric="auc",
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
        verbose=-1,
    )
    # 3) Grouped CV
    cv = RepeatedGroupKFoldWrapper(groups=groups, n_splits=5, n_repeats=3, random_state=42)
    # 4) Grid search
    print(" Starting LGBM Grid Search...")
    grid_search = GridSearchCV(
        lgbm_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)
    # 5) Best params/model
    best_params = grid_search.best_params_
    print(f"\n Best hyperparameters (LGBM):\n{best_params}")
    best_model_lgbm = lgb.LGBMClassifier(
        **best_params,
        boosting_type="gbdt",
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
        verbose=-1,
    )
    best_model_lgbm.fit(X_train_scaled, y_train)
    # 6) CV metrics (mean ± std)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    }
    cv_scores = cross_validate(
        best_model_lgbm, X_train_scaled, y_train,
        groups=groups, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    print("\n Cross-Validation (mean ± std):")
    cv_summary = {}
    for metric in scoring.keys():
        scores = cv_scores[f'test_{metric}']
        m, s = scores.mean(), scores.std()
        cv_summary[metric] = (m, s)
        print(f"{metric.capitalize():<10}: {m:.4f} ± {s:.4f}")
    # 7) Test set
    y_pred = best_model_lgbm.predict(X_test_scaled)
    y_pred_proba = best_model_lgbm.predict_proba(X_test_scaled)[:, 1]
    acc_test  = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test  = recall_score(y_test, y_pred, zero_division=0)
    f1_test   = f1_score(y_test, y_pred, zero_division=0)
    auc_test  = roc_auc_score(y_test, y_pred_proba)
    cm_test   = confusion_matrix(y_test, y_pred)

    print("\n Test Set Metrics (LGBM):")
    print(f"Accuracy:  {acc_test:.4f}")
    print(f"Precision: {prec_test:.4f}")
    print(f"Recall:    {rec_test:.4f}")
    print(f"F1 Score:  {f1_test:.4f}")
    print(f"ROC AUC:   {auc_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)

    # 8) Train set
    y_train_pred = best_model_lgbm.predict(X_train_scaled)
    y_train_pred_proba = best_model_lgbm.predict_proba(X_train_scaled)[:, 1]
    acc_tr  = accuracy_score(y_train, y_train_pred)
    prec_tr = precision_score(y_train, y_train_pred, zero_division=0)
    rec_tr  = recall_score(y_train, y_train_pred, zero_division=0)
    f1_tr   = f1_score(y_train, y_train_pred, zero_division=0)
    auc_tr  = roc_auc_score(y_train, y_train_pred_proba)
    cm_tr   = confusion_matrix(y_train, y_train_pred)

    print("\n Training Set Metrics (LGBM):")
    print(f"Accuracy:  {acc_tr:.4f}")
    print(f"Precision: {prec_tr:.4f}")
    print(f"Recall:    {rec_tr:.4f}")
    print(f"F1 Score:  {f1_tr:.4f}")
    print(f"ROC AUC:   {auc_tr:.4f}")
    print("Confusion Matrix:")
    print(cm_tr)

    # 9) Save model
    os.makedirs('classifier', exist_ok=True)
    model_path = os.path.join('classifier', 'lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_lgbm, f)
    print(f"\n Model saved to: {model_path}")
    # 10) Save plain-text report
    report_path = os.path.join('output', 'lgbm_report.txt')
    lines = []
    lines.append("LightGBM – Training Report\n")
    lines.append("Best Hyperparameters (Grid Search):")
    for k, v in best_params.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Cross-Validation (mean ± std)")
    for metric, (m, s) in cv_summary.items():
        lines.append(f"  {metric:>9}: {m:.4f} ± {s:.4f}")
    lines.append("")
    lines.append("Test Set Metrics")
    lines.append(f"  Accuracy : {acc_test:.4f}")
    lines.append(f"  Precision: {prec_test:.4f}")
    lines.append(f"  Recall   : {rec_test:.4f}")
    lines.append(f"  F1 Score : {f1_test:.4f}")
    lines.append(f"  ROC AUC  : {auc_test:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_test[0,0]}  {cm_test[0,1]}]")
    lines.append(f"     [{cm_test[1,0]}  {cm_test[1,1]}]]")
    lines.append("")
    lines.append("Training Set Metrics")
    lines.append(f"  Accuracy : {acc_tr:.4f}")
    lines.append(f"  Precision: {prec_tr:.4f}")
    lines.append(f"  Recall   : {rec_tr:.4f}")
    lines.append(f"  F1 Score : {f1_tr:.4f}")
    lines.append(f"  ROC AUC  : {auc_tr:.4f}")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("    [[TN  FP]")  
    lines.append("     [FN  TP]]")
    lines.append(f"    [[{cm_tr[0,0]}  {cm_tr[0,1]}]")
    lines.append(f"     [{cm_tr[1,0]}  {cm_tr[1,1]}]]")
    lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Metrics report saved to: {report_path}")


# =========================
# MODEL LOADER
# =========================

def upload_classifiers(save_dir: Path):
  
    """
    Load all '*.pkl' models from a directory into a dict.

    Parameters
    ----------
    save_dir : Path
        Directory containing pickled models (e.g., 'classifiers/').

    Returns
    -------
    dict
        {model_basename: estimator}, e.g. {'xgb': XGBClassifier(...), ...}
    """
  
    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory not found: {save_dir}")
    loaded_models = {}
    for model_file in os.listdir(save_dir):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace("_model.pkl", "")  # strip suffix
            model_path = save_dir / model_file
            with open(model_path, "rb") as f:
                loaded_models[model_name] = pickle.load(f)
            print(f" Loaded {model_name} from {model_path}")
    return loaded_models



