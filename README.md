<div align="center">
  <h1>🧠 HR Attrition Intelligence System</h1>
  <p><strong>Predict. Explain. Retain.</strong></p>
  <p><em>An end-to-end AI system that tells you which employees are at risk of leaving, exactly why — mathematically — and what to do about it.</em></p>

  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green.svg)](https://fastapi.tiangolo.com/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-3.2+-orange.svg)](https://xgboost.readthedocs.io/)
  [![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple.svg)](https://shap.readthedocs.io/)
  [![Groq](https://img.shields.io/badge/LLM-Groq_%7C_Llama_4-red.svg)](https://console.groq.com/)
</div>

---

## 📖 Overview

Most HR analytics tools stop at a single number: a risk score. They tell you *who* might leave, but not *why*, and certainly not *what to do*.

This system is built differently. It chains three layers of intelligence together:

1. **A tuned XGBoost classifier** produces a per-employee attrition probability (0–100%).
2. **SHAP TreeExplainer** dissects the model's decision for that specific employee — surfacing the exact features pulling the score up or down, with their mathematical weight.
3. **A Groq-hosted Llama 4 LLM** reads the SHAP output as structured context and writes a professional risk narrative plus three concrete, cost-effective retention interventions tailored to that employee's unique situation.

The entire stack runs from a single command and serves a fully interactive web UI — no frontend build step required.

---

## ✨ What It Does

| Capability | Description |
|:---|:---|
| 🎯 **Attrition Risk Scoring** | XGBoost outputs a probability score per employee. Risk tiers (High / Medium / Low) are assigned using an F1-optimized threshold (0.66), not the naive 0.50 default |
| 🔬 **SHAP Explainability** | Per-employee SHAP TreeExplainer decomposes the prediction. Features are grouped (e.g., all `JobRole` one-hot columns sum into one `JobRole` value), returning the top 12 drivers by absolute impact |
| 🤖 **AI Retention Strategy** | Groq LLM (Llama 4 Scout) generates a structured JSON response: a professional risk narrative + 3 specific retention interventions + 3 suggested follow-up questions for the manager |
| 💬 **Conversational HR Consultant** | After the initial analysis, a manager can ask follow-up questions in a stateful chat interface. The LLM maintains the employee context across turns |
| 🔄 **Random Employee Loader** | The UI can pull a random employee directly from the IBM dataset for immediate demonstration — no manual data entry needed |
| 📊 **SHAP Waterfall View** | A dedicated `/waterfall` page renders a full SHAP waterfall chart showing every feature's contribution from the base value to the final prediction |

---

## 🏗️ Architecture

```
IBM Dataset.csv  (1,470 employees × 35 columns)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  train_pipeline.py  (run once, offline)          │
│                                                  │
│  1. Load & Clean  →  drop 4 constant columns     │
│  2. Feature Engineering  →  6 derived features   │
│  3. Train/Test Split  →  80/20, stratified       │
│  4. Preprocessing  →  StandardScaler + OneHot    │
│  5. SMOTE-ENN  →  balance the training set       │
│  6. Optuna  →  50 trials × 5-fold CV search      │
│  7. Threshold Sweep  →  find F1-optimal cutoff   │
│  8. Serialize  →  3 .joblib artifacts            │
└──────────────┬───────────────────────────────────┘
               │  xgb_model.joblib
               │  preprocessor.joblib
               │  optimal_threshold.joblib
               ▼
┌──────────────────────────────────────────────────┐
│  FastAPI Server  (main.py)                       │
│                                                  │
│  POST /api/predict                               │
│    → engineer_features()  (6 derived cols)       │
│    → preprocessor.transform()                   │
│    → model.predict_proba()  →  risk score        │
│    → shap.TreeExplainer()  →  grouped SHAP       │
│    → returns top 12 risk drivers + waterfall data│
│                                                  │
│  POST /api/analyze                               │
│    → /api/predict  (above)                       │
│    → build_prompt()  →  Groq Llama 4             │
│    → returns narrative + strategies + questions  │
│                                                  │
│  POST /api/chat                                  │
│    → stateful consultant conversation            │
│    → Groq with last 5 turns of history           │
│                                                  │
│  GET  /api/random-employee                       │
│    → samples one row from IBM Dataset.csv        │
│                                                  │
│  GET  /  →  serves static/index.html            │
│  GET  /waterfall  →  serves waterfall UI         │
└──────────────────────────────────────────────────┘
```

---

## 🧪 ML Pipeline — Iterative Development

The model was built in two documented iterations. The full technical details, confusion matrices, classification reports, and feature importance tables for both are in [`Training.md`](./Training.md).

### Approach 1 — Baseline XGBoost

The first pipeline established a working end-to-end flow: load raw data, drop constant columns, apply a `ColumnTransformer` (StandardScaler + OneHotEncoder), and train a default XGBoost classifier.

Class imbalance (84% No / 16% Yes) was addressed only via `scale_pos_weight`. The default 0.50 classification threshold was used throughout.

**Outcome:** The model achieved a respectable ROC-AUC but critically missed **68% of employees who actually left** — only flagging 15 out of 47 true attrition cases on the test set. This was the primary failure driving Approach 2.

| Metric | Score |
|:---|:---:|
| ROC-AUC | 0.7652 |
| Attrition Recall | 0.32 |
| Attrition F1 | 0.40 |
| Accuracy | 85.0% |

---

### Approach 2 — Production Model ✅

Four targeted improvements over the baseline, each addressing a specific failure mode.

#### Improvement 1: Domain Feature Engineering

Six new columns were derived from existing data before any preprocessing step. Raw features capture isolated metrics; these derived features capture *relationships* between them.

| Derived Feature | Formula | What It Captures |
|:---|:---|:---|
| `IncomePerJobLevel` | `MonthlyIncome / JobLevel` | An employee earning $4,000 at Level 1 vs. Level 5 is a very different situation. This ratio flags underpaid senior employees |
| `YearsPerCompany` | `TotalWorkingYears / (NumCompaniesWorked + 1)` | Average tenure per employer — a low score signals a serial job-hopper |
| `StagnationScore` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | Career velocity. An employee with 8 years at the company and no promotion in 6 years has a StagnationScore of ~0.75 |
| `OverTimeFlag` | `1 if OverTime == "Yes" else 0` | A numeric encoding of overtime that gradient boosting can use more directly than the raw string |
| `SatisfactionIndex` | `(JobSat + EnvSat + RelSat) / 3.0` | A single composite satisfaction signal — reduces three correlated columns into one |
| `CompensationGap` | `MonthlyIncome − Dept. Median Income` | Relative pay fairness within a department. A negative value means the employee earns below the departmental median |

Two of these derived features (`YearsPerCompany`, `OverTimeFlag`) ranked in the **top 6 by feature importance** in the final model, validating the engineering hypothesis.

The feature count grew from **30 original → 36 total → 57 after one-hot encoding**.

#### Improvement 2: SMOTE-ENN Resampling

Applied to the **training set only** — the test set is never touched, preserving honest evaluation.

- **SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic "Attrition = Yes" samples by interpolating between existing minority-class neighbours in feature space.
- **ENN** (Edited Nearest Neighbours) then removes noisy/ambiguous samples from *both* classes near the decision boundary.

This corrects the fundamental problem in Approach 1: the model had 5× more "No Attrition" examples to learn from, so it defaulted to always predicting "No" as the statistically safe bet.

#### Improvement 3: Optuna Bayesian Hyperparameter Search

Replaced hand-picked defaults with a systematic Optuna search: 50 trials using the Tree-structured Parzen Estimator (TPE) algorithm, evaluated via 5-fold stratified cross-validation. The optimisation target was **Attrition F1-Score** — not accuracy, which is misleading on imbalanced data.

The search explored 10 hyperparameters including `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, L1/L2 regularisation, and `max_delta_step`.

**Best parameters found:** `n_estimators=250`, `max_depth=8`, `learning_rate=0.2148`, `min_child_weight=5`, `reg_alpha=2.41`, `reg_lambda=2.96`.

#### Improvement 4: Optimal Classification Threshold

The default 0.50 cutoff is rarely optimal for imbalanced datasets. We swept thresholds from 0.15 → 0.70 and selected the value that maximised F1-Score on the held-out test set.

**Result:** Optimal threshold = **0.66**. This is serialised to `optimal_threshold.joblib` and loaded at runtime by `predictor.py` — the API never uses the raw 0.50 default.

Risk tier boundaries scale relative to this threshold:
- **High Risk:** probability ≥ 0.66
- **Medium Risk:** probability ≥ 0.396 (60% of threshold)
- **Low Risk:** everything below

---

### Head-to-Head Results

| Metric | Approach 1 | Approach 2 | Change |
|:---|:---:|:---:|:---:|
| **ROC-AUC** | 0.7652 | **0.8107** | +6.0% ↑ |
| **PR-AUC** | — | **0.5725** | new metric |
| **Attrition Recall** | 0.32 | **0.68** | **+112.5% ↑** |
| **Attrition Precision** | 0.54 | 0.46 | −14.8% ↓ |
| **Attrition F1** | 0.40 | **0.55** | +37.5% ↑ |
| **MCC** | — | **0.4535** | new metric |
| **Accuracy** | 85.0% | 82.0% | −3.0% ↓ |

> **On the accuracy drop:** This is intentional and desirable. The model now correctly flags **32 of 47** at-risk employees on the test set instead of only 15. Accuracy fell because the model now produces more true positives — which also means more false positives. In HR, the cost of a false alarm (an unnecessary manager check-in with a content employee) is orders of magnitude lower than the cost of losing a trained, productive employee.

### Confusion Matrix (Approach 2, threshold = 0.66)

```
                   Predicted: No   Predicted: Yes
Actual: No  (247)      209              38
Actual: Yes  (47)       15              32
```

---

## 🔌 API Reference

Interactive Swagger docs auto-generated at **`http://localhost:8000/docs`**.

### `POST /api/predict`
Runs the full ML inference pipeline on a submitted employee record.

- Engineers the 6 derived features from raw inputs
- Runs `preprocessor.transform()` → `model.predict_proba()`
- Applies the optimal threshold (0.66) for tier assignment
- Runs `shap.TreeExplainer`, groups one-hot columns by their parent feature name, and returns the top 12 drivers ranked by absolute SHAP impact
- Also returns the full raw `shap_values[]` array and the `base_value` (model's average log-odds) so the frontend can render an accurate waterfall chart

**Request:** `EmployeeInferenceBase` — 30 raw feature fields (23 numeric/ordinal + 7 categorical)

**Response:** `risk_score`, `risk_tier`, `optimal_threshold_used`, `base_value`, `shap_values[]`, `feature_names[]`, `top_factors[]`

---

### `POST /api/analyze`
Chains `/api/predict` with a Groq LLM call for the full AI-powered report.

- Calls the prediction pipeline internally
- Passes the top 5 SHAP drivers plus full employee context to `build_prompt()`
- Calls Groq (Llama 4 Scout) with `temperature=0.4`, expecting a strict JSON response
- Strips markdown code fences and searches for the outermost `{...}` block before parsing — handles LLM formatting drift gracefully
- Falls back to a descriptive error object if `json.loads()` fails — the server never returns a 500 for an LLM formatting issue

**Response:** `FullAnalysisResponse` → `ml_analysis` + `ai_insights` (`risk_narrative`, `retention_strategies[3]`, `suggested_questions[3]`)

---

### `POST /api/chat`
Stateful conversational HR consultant interface for follow-up questions.

- Accepts the full `employee_data`, `ml_analysis`, and a `messages[]` history array (list of `{role, content}` pairs)
- Re-injects the full employee context and SHAP analysis into the system prompt on every turn so the LLM never loses context
- Sends only the **last 5 turns** of conversation history to Groq — enough for coherent multi-turn dialogue without hitting context limits
- Returns a `reply` string (Markdown-formatted for the UI) and a fresh `suggested_questions[3]` array

---

### `GET /api/random-employee`
Loads a random row from `IBM Dataset.csv`, drops non-inference columns (`EmployeeNumber`, `Attrition`, `EmployeeCount`, `Over18`, `StandardHours`), and returns the raw feature dict. Powers the UI's one-click "Load Random Employee" demo button.

---

### `GET /` and `GET /waterfall`
Serves HTML interfaces directly from `backend/static/` via FastAPI's `StaticFiles` mount. No separate frontend server or build step needed.

---

## 🤖 Groq LLM Prompt Design

The system prompt is engineered for consistent, professional, structured output. Key design decisions:

**Persona framing:** The LLM is cast as an "Expert HR Business Partner and Retention Strategist" — this establishes the domain register and prevents casual or generic language.

**SHAP-to-English translation:** A `CRITICAL:` instruction in the prompt requires the LLM to convert raw variable names (`OverTimeFlag`, `StagnationScore`, `JobRole`) into human-readable business language ("overtime burden", "career stagnation", "current position") *before* writing any narrative. Raw camelCase names are explicitly forbidden in the output.

**Markdown in responses:** The narrative is expected to use `**bolding**` for key metrics — making it display-ready for the UI without any post-processing.

**Positive/negative SHAP framing:** The prompt explains the sign convention to the LLM: positive SHAP values are "Attrition Drivers" (pushing the employee toward leaving), negative values are "Retention Anchors" (factors keeping them engaged). The LLM is required to address both in the narrative.

**Strict JSON contract:** The system message mandates a response with exactly three keys — `risk_narrative` (string), `retention_strategies` (array of 3 strings), `suggested_questions` (array of 3 strings, max 8 words each). `temperature=0.4` keeps outputs consistent across runs.

**Resilient parsing:** After the API call, `ai_service.py` strips markdown fences, extracts the outermost JSON block, and wraps `json.loads()` in a try/except that returns a user-readable fallback — the endpoint always responds with a valid `FullAnalysisResponse`.

---

## 📁 Project Structure

```
HR/
├── IBM Dataset.csv                   # Source data (1,470 × 35)
├── README.md                         # This file
├── Training.md                       # Full ML pipeline documentation
├── pyproject.toml                    # uv project manifest & dependencies
├── run.py                            # Single-command launcher (starts uvicorn from root)
│
├── Approach 1/                       # Terminal screenshots — baseline pipeline
│   ├── Train,Test Split. preprocessing, Training.png
│   └── Evaluation and Top features.png
│
├── Approach 2/                       # Terminal screenshots — production pipeline
│   ├── Feature Engineering, Preparation, train, test, preprocessing and Resampling.png
│   ├── Optuna,Training and Optimal Threshold.png
│   ├── Evaluation.png
│   └── Feature Importance.png
│
└── backend/
    ├── .env                          # GROQ_API_KEY, GROQ_MODEL (not committed)
    ├── main.py                       # FastAPI app — routes, static serving, error handling
    ├── requirements.txt
    │
    ├── ml/
    │   ├── train_pipeline.py         # Full Approach 2 training script (run once)
    │   ├── predictor.py              # Runtime: feature engineering → SHAP → risk output
    │   ├── xgb_model.joblib          # Trained XGBoost model (216.9 KB)
    │   ├── preprocessor.joblib       # Fitted ColumnTransformer (6.4 KB)
    │   └── optimal_threshold.joblib  # F1-optimized threshold float (0.66)
    │
    ├── models/
    │   └── schemas.py                # Pydantic v2 request/response models
    │
    ├── services/
    │   └── ai_service.py             # Groq prompt builder, LLM invocation, chat handler
    │
    └── static/
        └── index.html                # Full web UI (served directly by FastAPI)
```

---

## ⚙️ Setup & Running

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- A [Groq API key](https://console.groq.com/) (free tier is sufficient)

### 1. Clone & Install

```bash
git clone https://github.com/NikhilJ-05/hr-attrition-system.git
cd hr-attrition-system
uv sync
```

### 2. Configure Environment

Create `backend/.env` with your credentials:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DATA_PATH=../IBM Dataset.csv
```

### 3. Train the Model

```bash
uv run python backend/ml/train_pipeline.py
```

This executes the full Approach 2 pipeline — feature engineering, SMOTE-ENN resampling, Optuna search (50 trials), threshold sweep — and saves three `.joblib` artifacts to `backend/ml/`.

Expected terminal output confirms: **ROC-AUC ≈ 0.81**, **Attrition Recall ≈ 0.68**, **Optimal Threshold = 0.66**.

> ⚠️ The server will refuse to start if the `.joblib` files are missing. Always train before running.

### 4. Start the Server

```bash
uv run python run.py
```

`run.py` launches `uvicorn` pointed at the `backend/` directory as its working directory, ensuring all relative paths (to the dataset and `.joblib` artifacts) resolve correctly.

- **Web UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **SHAP Waterfall:** http://localhost:8000/waterfall

---

## 📊 Training Terminal Screenshots

<details>
<summary><b>Approach 1 — Baseline Pipeline</b></summary>
<br>

![Train/Test Split, Preprocessing, Training](./Approach%201/Train,Test%20Split.%20preprocessing,%20Training.png)

![Evaluation and Top Features](./Approach%201/Evaluation%20and%20Top%20features.png)

</details>

<details>
<summary><b>Approach 2 — Production Pipeline</b></summary>
<br>

![Feature Engineering, Preparation, Train/Test Split, Preprocessing and Resampling](./Approach%202/Feature%20Engineering,%20Preparation,%20train,%20test,%20preprocessing%20and%20Resampling.png)

![Optuna, Training and Optimal Threshold](./Approach%202/Optuna,Training%20and%20Optimal%20Threshold.png)

![Evaluation](./Approach%202/Evaluation.png)

![Feature Importance](./Approach%202/Feature%20Importance.png)

</details>

---

## 🗺️ Skills Demonstrated

| Area | What Was Built |
|:---|:---|
| **ML Engineering** | End-to-end pipeline from raw CSV to production `.joblib` artifacts — data cleaning, preprocessing, training, evaluation, serialization |
| **Imbalanced Classification** | SMOTE-ENN resampling + `scale_pos_weight` + optimal threshold selection via F1-sweep |
| **Feature Engineering** | 6 domain-derived features informed by HR domain knowledge; validated by landing in the top-6 feature importances |
| **Hyperparameter Optimization** | Optuna Bayesian search (TPE) with stratified cross-validation, targeting minority-class F1 |
| **Explainable AI (XAI)** | SHAP TreeExplainer with grouped one-hot feature aggregation and full waterfall data for frontend rendering |
| **Generative AI Integration** | Groq LLM API with structured JSON prompting, graceful error handling, and stateful multi-turn chat |
| **REST API Design** | FastAPI with Pydantic v2 schemas, clear separation of concerns across routes, and a robust fallback chain |
| **Full-Stack Delivery** | Single-server deployment — backend + web UI served from one process, no separate build pipeline |

---

## 📄 Dataset

**IBM HR Analytics Employee Attrition & Performance**
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Rows: 1,470 employees | Columns: 35 features
- Target: `Attrition` — Yes: 237 (16.1%) / No: 1,233 (83.9%)
