import os
import joblib
import pandas as pd
import shap

# Load artifacts once at startup
ML_DIR = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ML_DIR, 'xgb_model.joblib')
PREPROCESSOR_PATH = os.path.join(ML_DIR, 'preprocessor.joblib')
THRESHOLD_PATH = os.path.join(ML_DIR, 'optimal_threshold.joblib')

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    # If the threshold file doesn't exist (e.g., Approach 1), fallback to 0.50
    try:
        optimal_threshold = joblib.load(THRESHOLD_PATH)
    except FileNotFoundError:
        optimal_threshold = 0.50
except FileNotFoundError as e:
    raise RuntimeError(f"ML artifacts not found. Please train the model first. Error: {e}")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same 6 domain features used during training."""
    df = df.copy()
    df['IncomePerJobLevel'] = df['MonthlyIncome'] / df['JobLevel']
    df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    df['StagnationScore'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['OverTimeFlag'] = (df['OverTime'] == 'Yes').astype(int)
    df['SatisfactionIndex'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 3.0
    
    # In production, we should ideally use the actual department medians from the training set.
    # For this interim interface, we approximate on the fly or use a static map.
    dept_medians = {
        'Sales': 5754.5, 
        'Research & Development': 4374.0, 
        'Human Resources': 3886.0
    }
    df['CompensationGap'] = df.apply(lambda row: row['MonthlyIncome'] - dept_medians.get(row['Department'], 4500), axis=1)
    
    return df

def predict_attrition(employee_data: dict) -> dict:
    """
    Takes raw employee features, engineers them, preprocesses,
    predicts risk score, assigns a tier based on the optimized threshold,
    and returns SHAP explanations with grouped features and waterfall data.
    """
    # 1. Convert to DataFrame (single row)
    df = pd.DataFrame([employee_data])

    # 2. Engineer features
    df_engineered = engineer_features(df)

    # 3. Preprocess
    X_processed = preprocessor.transform(df_engineered)

    # 4. Predict probability
    proba = model.predict_proba(X_processed)[0][1]
    risk_score = float(proba) * 100.0

    # 5. Risk Tier Assignment
    if proba >= optimal_threshold:
        risk_tier = "High"
    elif proba >= (optimal_threshold * 0.6):
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # 6. SHAP Explanations
    shap_values = explainer.shap_values(X_processed)
    feature_names = preprocessor.get_feature_names_out()
    base_value = float(explainer.expected_value)  # Base log-odds (average model output)

    # Build a mapping of grouped features
    # Map: parent_feature -> list of (full_feature_name, shap_value)
    grouped_shap = {}

    for feat_name, shap_val in zip(feature_names, shap_values[0]):
        # Clean up sklearn's transformer prefixes
        clean_name = feat_name.replace("num__", "").replace("cat__", "")

        # Identify parent feature for grouping
        # One-hot features have format: Parent_Child or Parent=Child
        parent_feature = clean_name.split("_")[0] if "_" in clean_name else clean_name.split("=")[0] if "=" in clean_name else clean_name

        # Special handling for derived features: keep them as their own group
        if parent_feature in ['IncomePerJobLevel', 'YearsPerCompany', 'StagnationScore',
                              'OverTimeFlag', 'SatisfactionIndex', 'CompensationGap']:
            parent_feature = clean_name  # Use full name as group

        if parent_feature not in grouped_shap:
            grouped_shap[parent_feature] = []
        grouped_shap[parent_feature].append((clean_name, shap_val))

    # Aggregate groups: sum absolute impacts, track direction
    group_aggregates = []
    for parent, contributions in grouped_shap.items():
        total_impact = float(sum(shap_val for _, shap_val in contributions))
        # Determine direction from sign of first significant contributor
        primary_sign = 1 if any(shap_val > 0 for _, shap_val in contributions) else -1
        abs_impact = abs(total_impact)
        group_aggregates.append({
            'feature': parent,
            'impact': total_impact,
            'abs_impact': abs_impact,
            'raw_components': contributions  # Keep for potential future use
        })

    # Sort by absolute impact and take top 12 (more than 5 for richer demo)
    group_aggregates.sort(key=lambda x: x['abs_impact'], reverse=True)
    top_groups = group_aggregates[:12]

    # Format for API response
    top_factors = []
    for group in top_groups:
        direction = "+" if group['impact'] > 0 else "-"
        top_factors.append({
            "feature": group['feature'],
            "impact": f"{direction}{group['abs_impact']:.2f}",
            "raw_impact": float(round(group['impact'], 4))  # For frontend calculations
        })

    # Convert shap_values to plain list for JSON serialization (waterfall data)
    shap_list = [float(val) for val in shap_values[0]]

    return {
        "risk_score": round(risk_score, 1),
        "risk_tier": risk_tier,
        "optimal_threshold_used": float(optimal_threshold),
        "base_value": round(float(base_value), 4),  # For waterfall starting point
        "shap_values": shap_list,  # Full array for waterfall visualization
        "feature_names": [str(name) for name in feature_names],  # Corresponding feature names
        "top_factors": top_factors
    }
