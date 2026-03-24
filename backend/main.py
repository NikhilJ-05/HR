from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from models.schemas import EmployeeInferenceBase, PredictionResponse, FullAnalysisResponse, MLAnalysis, AIInsights, RiskDriver, ChatRequest
from ml.predictor import predict_attrition
from services.ai_service import generate_insights, chat_with_consultant

app = FastAPI(title="HR Attrition Predictor (Interim UI)")

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the Interim HTML Testing Interface"""
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
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
    Takes raw employee features, engineers them, preprocesses,
    predicts risk (tuned threshold=0.66), returns SHAP explanations.
    """
    try:
        raw_data = employee.dict()
        result = predict_attrition(raw_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=FullAnalysisResponse)
async def analyze_employee(employee: EmployeeInferenceBase):
    """
    Runs full analysis: ML prediction + SHAP + Groq LLM insights.
    Returns combined result with risk narrative and retention strategies.
    """
    try:
        raw_data = employee.dict()
        ml_result = predict_attrition(raw_data)

        # Build MLAnalysis
        top_drivers = [
            RiskDriver(
                feature=f["feature"],
                impact=f["impact"],
                raw_impact=f.get("raw_impact")
            )
            for f in ml_result.get("top_factors", [])
        ]
        ml_analysis = MLAnalysis(
            risk_score=ml_result["risk_score"],
            risk_tier=ml_result["risk_tier"],
            optimal_threshold_used=ml_result["optimal_threshold_used"],
            base_value=ml_result["base_value"],
            shap_values=ml_result["shap_values"],
            feature_names=ml_result["feature_names"],
            top_drivers=top_drivers
        )

        # Generate AI insights via Groq
        try:
            ai_dict = generate_insights(raw_data, ml_result)
            ai_insights = AIInsights(**ai_dict)
        except Exception as e:
            # Log to server console
            print(f"Groq API error: {e}")
            ai_insights = AIInsights(
                risk_narrative="AI insight generation failed. Please check Groq API key and connectivity.",
                retention_strategies=[],
                suggested_questions=["Why did insight generation fail?", "How can I fix the API connection?"]
            )

        return FullAnalysisResponse(
            employee_number=raw_data.get("EmployeeNumber"),
            ml_analysis=ml_analysis,
            ai_insights=ai_insights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/waterfall", response_class=HTMLResponse)
async def waterfall_page():
    """Serves the SHAP waterfall demo page"""
    with open(os.path.join(STATIC_DIR, "waterfall.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response_data = chat_with_consultant(
            employee_data=request.employee_data,
            ml_analysis=request.ml_analysis,
            messages=request.messages
        )
        return {
            "reply": response_data.get("reply", "I am unable to process that request at this moment."),
            "suggested_questions": response_data.get("suggested_questions", [])
        }
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
