import pandas as pd
from ml.predictor import predict_attrition

# Load dataset
csv_path = "../IBM Dataset.csv"
df = pd.read_csv(csv_path)

# Get a random employee and clean to API format
employee = df.sample(1).to_dict(orient='records')[0]

# Remove fields that the API would not send
for col in ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']:
    employee.pop(col, None)

print("Employee sample:", employee.get('Department'), employee.get('JobRole'), employee.get('OverTime'))

result = predict_attrition(employee)

print("\n=== Enhanced SHAP Output ===")
print(f"Risk Score: {result['risk_score']}%")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Base Value: {result['base_value']}")
print(f"Optimal Threshold: {result['optimal_threshold_used']}")
print("\nTop Factors (Grouped, Top 12):")
for i, f in enumerate(result['top_factors'], 1):
    print(f"{i:2}. {f['feature']:25} {f['impact']}")
print(f"\nSHAP array length: {len(result['shap_values'])}")
print(f"Feature names count: {len(result['feature_names'])}")
