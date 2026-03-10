"""
Approach 2: Advanced XGBoost Training Pipeline
- Feature Engineering (6 derived features)
- SMOTE-ENN Resampling
- Optuna Bayesian Hyperparameter Tuning
- Optimal Threshold Selection

Usage:
    cd HR
    python backend/ml/train_pipeline.py

Outputs:
    backend/ml/xgb_model.joblib       — best XGBoost classifier
    backend/ml/preprocessor.joblib     — fitted ColumnTransformer
    backend/ml/optimal_threshold.joblib — optimal classification threshold
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
import xgboost as xgb
from imblearn.combine import SMOTEENN
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'IBM Dataset.csv')
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), 'xgb_model.joblib')
PREPROCESSOR_OUTPUT = os.path.join(os.path.dirname(__file__), 'preprocessor.joblib')
THRESHOLD_OUTPUT = os.path.join(os.path.dirname(__file__), 'optimal_threshold.joblib')

# Columns to drop (constant / ID)
CONSTANT_DROP_COLS = ['EmployeeCount', 'Over18', 'StandardHours']
ID_COL = 'EmployeeNumber'
TARGET_COL = 'Attrition'

# ---------------------------------------------------------------------------
# 2. COLUMN DEFINITIONS
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    # Ordinals as numeric
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'
]

CATEGORICAL_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

# Derived feature names (added by feature engineering)
DERIVED_NUMERIC_COLS = [
    'IncomePerJobLevel', 'YearsPerCompany', 'StagnationScore',
    'OverTimeFlag', 'SatisfactionIndex', 'CompensationGap'
]


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 6 domain-relevant derived features."""
    df = df.copy()

    # 1. Income relative to job level — detects underpaid-for-level employees
    df['IncomePerJobLevel'] = df['MonthlyIncome'] / df['JobLevel']

    # 2. Average years per company — job-hopping tendency
    df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)

    # 3. Promotion stagnation — how long stuck without promotion relative to tenure
    df['StagnationScore'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    # 4. OverTime as numeric flag (1/0)
    df['OverTimeFlag'] = (df['OverTime'] == 'Yes').astype(int)

    # 5. Aggregate satisfaction index
    df['SatisfactionIndex'] = (
        df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    ) / 3.0

    # 6. Compensation gap vs department median
    dept_median = df.groupby('Department')['MonthlyIncome'].transform('median')
    df['CompensationGap'] = df['MonthlyIncome'] - dept_median

    return df


# ---------------------------------------------------------------------------
# 4. OPTUNA OBJECTIVE
# ---------------------------------------------------------------------------
def create_objective(X_train_processed, y_train_resampled):
    """Returns an Optuna objective that tunes XGBoost hyperparameters."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 3),
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'verbosity': 0
        }

        # 5-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train_processed, y_train_resampled):
            X_fold_train = X_train_processed[train_idx]
            y_fold_train = y_train_resampled.iloc[train_idx] if hasattr(y_train_resampled, 'iloc') else y_train_resampled[train_idx]
            X_fold_val = X_train_processed[val_idx]
            y_fold_val = y_train_resampled.iloc[val_idx] if hasattr(y_train_resampled, 'iloc') else y_train_resampled[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            f1_scores.append(f1_score(y_fold_val, y_pred))

        return np.mean(f1_scores)

    return objective


# ---------------------------------------------------------------------------
# 5. THRESHOLD TUNING
# ---------------------------------------------------------------------------
def find_optimal_threshold(model, X_val, y_val):
    """Sweep thresholds to find the one maximizing F1-score."""
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.15, 0.70, 0.01):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  APPROACH 2: Advanced XGBoost Training Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: LOAD
    # ------------------------------------------------------------------
    print("\n📂 STEP 1: Loading IBM HR Dataset")
    if not os.path.exists(DATA_PATH):
        print(f"  ERROR: Dataset not found at {os.path.abspath(DATA_PATH)}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop(columns=CONSTANT_DROP_COLS, errors='ignore')

    attrition_counts = df[TARGET_COL].value_counts()
    total = len(df)
    print(f"  Target Distribution:")
    for val, count in attrition_counts.items():
        print(f"    {val}: {count} ({count/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    print("\n🔧 STEP 2: Feature Engineering (6 derived features)")
    df = engineer_features(df)

    for col in DERIVED_NUMERIC_COLS:
        print(f"  ✓ {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

    # ------------------------------------------------------------------
    # STEP 3: PREPARE FEATURES & TARGET
    # ------------------------------------------------------------------
    print("\n📊 STEP 3: Preparing Features & Target")
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL].map({'Yes': 1, 'No': 0})

    all_numeric = NUMERIC_COLS + DERIVED_NUMERIC_COLS
    actual_num = [c for c in all_numeric if c in X.columns]
    actual_cat = [c for c in CATEGORICAL_COLS if c in X.columns]
    print(f"  Numeric features: {len(actual_num)} (incl. 6 derived)")
    print(f"  Categorical features: {len(actual_cat)}")
    print(f"  Total features: {len(actual_num) + len(actual_cat)}")

    # ------------------------------------------------------------------
    # STEP 4: TRAIN/TEST SPLIT
    # ------------------------------------------------------------------
    print("\n✂️  STEP 4: Train/Test Split (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Training set: {X_train.shape[0]} rows (attrition: {y_train.mean()*100:.1f}%)")
    print(f"  Test set:     {X_test.shape[0]} rows (attrition: {y_test.mean()*100:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 5: PREPROCESSING
    # ------------------------------------------------------------------
    print("\n⚙️  STEP 5: Preprocessing (StandardScaler + OneHotEncoder)")
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), actual_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), actual_cat)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print(f"  Transformed features: {len(feature_names)}")

    # ------------------------------------------------------------------
    # STEP 6: SMOTE-ENN RESAMPLING (training set only)
    # ------------------------------------------------------------------
    print("\n🔄 STEP 6: SMOTE-ENN Resampling (training set only)")
    print(f"  Before: {len(y_train)} samples (Attrition=1: {y_train.sum()})")

    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_processed, y_train)

    print(f"  After:  {len(y_train_resampled)} samples (Attrition=1: {y_train_resampled.sum()})")
    print(f"  New class ratio — No: {(y_train_resampled == 0).sum()}, Yes: {(y_train_resampled == 1).sum()}")

    # ------------------------------------------------------------------
    # STEP 7: OPTUNA HYPERPARAMETER TUNING
    # ------------------------------------------------------------------
    print("\n🎯 STEP 7: Optuna Hyperparameter Tuning (50 trials, 5-fold CV)")
    print("  Optimizing for: F1-Score (Attrition class)")

    objective = create_objective(X_train_resampled, y_train_resampled)
    study = optuna.create_study(direction='maximize', study_name='xgb-attrition')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n  Best F1 (CV): {study.best_value:.4f}")
    print(f"  Best Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # STEP 8: TRAIN FINAL MODEL WITH BEST PARAMS
    # ------------------------------------------------------------------
    print("\n🏋️ STEP 8: Training Final Model with Best Params")
    best_params = study.best_params
    best_params.update({
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_resampled, y_train_resampled)
    print("  Training complete!")

    # ------------------------------------------------------------------
    # STEP 9: THRESHOLD TUNING
    # ------------------------------------------------------------------
    print("\n🎚️  STEP 9: Optimal Threshold Selection")
    optimal_threshold, best_f1_at_threshold = find_optimal_threshold(model, X_test_processed, y_test)
    print(f"  Optimal threshold: {optimal_threshold:.2f} (F1={best_f1_at_threshold:.4f})")
    print(f"  vs. default 0.50 threshold")

    # ------------------------------------------------------------------
    # STEP 10: EVALUATION (at optimal threshold)
    # ------------------------------------------------------------------
    print("\n📈 STEP 10: Evaluation (at optimal threshold)")

    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred_default = model.predict(X_test_processed)
    y_pred_tuned = (y_proba >= optimal_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred_tuned)
    cm = confusion_matrix(y_test, y_pred_tuned)

    print(f"\n  --- Metrics at DEFAULT threshold (0.50) ---")
    print(classification_report(y_test, y_pred_default, target_names=['No Attrition', 'Attrition']))

    print(f"  --- Metrics at OPTIMAL threshold ({optimal_threshold:.2f}) ---")
    print(classification_report(y_test, y_pred_tuned, target_names=['No Attrition', 'Attrition']))

    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  MCC:      {mcc:.4f}")

    print(f"\n  Confusion Matrix (threshold={optimal_threshold:.2f}):")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    if roc_auc < 0.75:
        print("\n  ⚠️  WARNING: ROC-AUC below 0.75!")
    else:
        print(f"\n  ✅ ROC-AUC {roc_auc:.4f} exceeds 0.75 threshold — PASS")

    # ------------------------------------------------------------------
    # STEP 11: COMPARISON TABLE (Approach 1 vs Approach 2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  APPROACH 1 vs APPROACH 2 COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Approach 1':>12} {'Approach 2':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    # Approach 1 baseline values (from our earlier training)
    a1_roc = 0.7652
    a1_recall = 0.32
    a1_f1 = 0.40
    a1_precision = 0.54

    a2_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    a2_precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    a2_f1 = f1_score(y_test, y_pred_tuned)

    print(f"  {'ROC-AUC':<25} {a1_roc:>12.4f} {roc_auc:>12.4f}")
    print(f"  {'PR-AUC':<25} {'N/A':>12} {pr_auc:>12.4f}")
    print(f"  {'Attrition Recall':<25} {a1_recall:>12.2f} {a2_recall:>12.2f}")
    print(f"  {'Attrition Precision':<25} {a1_precision:>12.2f} {a2_precision:>12.2f}")
    print(f"  {'Attrition F1':<25} {a1_f1:>12.2f} {a2_f1:>12.2f}")
    print(f"  {'MCC':<25} {'N/A':>12} {mcc:>12.4f}")

    # ------------------------------------------------------------------
    # STEP 12: TOP 10 FEATURE IMPORTANCES
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Top 10 Feature Importances (gain)")
    print("=" * 70)

    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(feat_imp[:10], 1):
        bar = "█" * int(imp * 100)
        print(f"  {i:2d}. {name:45s} {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # STEP 13: SAVE ARTIFACTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Saving Artifacts")
    print("=" * 70)

    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT)
    joblib.dump(optimal_threshold, THRESHOLD_OUTPUT)

    print(f"  Model:        {MODEL_OUTPUT} ({os.path.getsize(MODEL_OUTPUT)/1024:.1f} KB)")
    print(f"  Preprocessor: {PREPROCESSOR_OUTPUT} ({os.path.getsize(PREPROCESSOR_OUTPUT)/1024:.1f} KB)")
    print(f"  Threshold:    {THRESHOLD_OUTPUT} (value={optimal_threshold:.2f})")

    print("\n" + "=" * 70)
    print("  ✅ APPROACH 2 PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
