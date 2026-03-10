from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from models.schemas import EmployeeInferenceBase, PredictionResponse
from ml.predictor import predict_attrition

app = FastAPI(title="HR Attrition Predictor (Interim UI)")

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the Interim HTML Testing Interface"""
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

@app.get("/api/random-employee")
async def get_random_employee():
    """Returns a single random employee record from the raw IBM dataset"""
    import pandas as pd
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "IBM Dataset.csv")
        df = pd.read_csv(csv_path)
        # Drop columns not expected by the inference schema
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Get one random row
        random_row = df.sample(1).to_dict(orient='records')[0]
        return random_row
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_employee(employee: EmployeeInferenceBase):
    """
    Takes 30 raw features, engineers 6 more, preprocesses, 
    predicts risk (tuned threshold=0.66), and calculates top 5 SHAP factors.
    """
    try:
        # Convert Pydantic to pure dictionary
        raw_data = employee.dict()
        
        # Run prediction pipeline
        result = predict_attrition(raw_data)
        
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
