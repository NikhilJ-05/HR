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
    and returns the top 5 driving factors via SHAP.
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
    
    # 5. Risk Tier Assignment (using optimal threshold instead of generic 0.5)
    # Using 66% as High, 40-66% as Medium, <40% as Low based on our Approach 2 threshold
    if proba >= optimal_threshold:
        risk_tier = "High"
    elif proba >= (optimal_threshold * 0.6): # e.g. 0.40 if threshold is 0.66
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    # 6. SHAP Explanations
    shap_values = explainer.shap_values(X_processed)
    feature_names = preprocessor.get_feature_names_out()
    
    shap_dict = dict(zip(feature_names, shap_values[0]))
    # Sort by absolute impact
    sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    top_factors = []
    for feat_name, impact in sorted_factors:
        # Clean up sklearn's transformer prefixes
        clean_name = feat_name.replace("num__", "").replace("cat__", "").replace("_", "=", 1)
        direction = "+" if impact > 0 else "-"
        # Convert raw log-odds impact to a relative percentage scale for UI display
        top_factors.append({
            "feature": clean_name,
            "impact": f"{direction}{abs(impact):.2f}"
        })
        
    return {
        "risk_score": round(risk_score, 1),
        "risk_tier": risk_tier,
        "optimal_threshold_used": optimal_threshold,
        "top_factors": top_factors
    }
