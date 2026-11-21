"""FastAPI Application for ChurnSense: End-to-end churn prediction, explainability, and UI serving."""

from contextlib import asynccontextmanager
import json
import logging
from typing import Literal, Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    PIPELINE_PATH,
    THRESHOLD_PATH,
    DEFAULT_THRESHOLD,
    MODELS_DIR,
    WEB_DIR,
)
from src.explain import ChurnExplainer
from src.llm_explain import explain
from src.chat import ask_churn_assistant

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("churnsense.api")

# Available model architectures
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "subtitle": "Tuned L2 Regularization (Production Active)",
        "file": MODELS_DIR / "pipeline_logistic_regression.joblib",
        "fallback_file": PIPELINE_PATH,
        "cv_roc_auc": "0.8460 ± 0.012",
        "test_roc_auc": 0.8408,
        "test_recall": "78.34%",
        "test_precision": "50.34%",
        "test_f1": "0.6130",
        "status": "Production Active",
        "badge": "Winner",
        "description": "Linear decision boundary with balanced class weights. Maximizes recall at calibrated 0.50 threshold.",
    },
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "subtitle": "300 Balanced Trees Ensemble",
        "file": MODELS_DIR / "pipeline_random_forest.joblib",
        "fallback_file": None,
        "cv_roc_auc": "0.8228 ± 0.012",
        "test_roc_auc": 0.8226,
        "test_recall": "63.37%",
        "test_precision": "54.73%",
        "test_f1": "0.5874",
        "status": "Benchmarked",
        "badge": "Ensemble",
        "description": "Non-linear bagging ensemble of 300 decision trees. Robust to feature interactions with higher precision.",
    },
    "xgboost": {
        "id": "xgboost",
        "name": "XGBoost Classifier",
        "subtitle": "Gradient Boosted Trees (scale_pos_weight=2.76)",
        "file": MODELS_DIR / "pipeline_xgboost.joblib",
        "fallback_file": None,
        "cv_roc_auc": "0.8246 ± 0.009",
        "test_roc_auc": 0.8187,
        "test_recall": "66.31%",
        "test_precision": "52.77%",
        "test_f1": "0.5877",
        "status": "Benchmarked",
        "badge": "Boosting",
        "description": "Sequential gradient boosting with logloss objective. Sensitive to subtle interaction boundaries.",
    },
}

# Global model state
state: Dict[str, Any] = {
    "pipeline": None,
    "pipelines": {},
    "threshold": DEFAULT_THRESHOLD,
    "explainer": None,
    "explainers": {},
}


def load_model_artifacts():
    """Load model pipelines, threshold, and SHAP explainers."""
    if state["pipeline"] is not None and len(state["pipelines"]) > 0:
        return

    logger.info("Initializing ChurnSense service artifacts...")
    if not PIPELINE_PATH.exists():
        logger.error(f"Pipeline artifact missing at {PIPELINE_PATH}. Please run training first.")
        raise RuntimeError("Trained model pipeline not found. Run training before launching API.")

    default_pipe = joblib.load(PIPELINE_PATH)
    state["pipeline"] = default_pipe
    state["pipelines"]["logistic_regression"] = default_pipe

    # Load other candidate models if available
    for m_id, meta in MODEL_REGISTRY.items():
        if m_id == "logistic_regression":
            continue
        p_path = meta["file"]
        if p_path.exists():
            try:
                state["pipelines"][m_id] = joblib.load(p_path)
                logger.info(f"Loaded {meta['name']} pipeline from {p_path.name}")
            except Exception as e:
                logger.warning(f"Could not load {m_id} pipeline: {e}")

    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            state["threshold"] = float(t_data.get("optimal_threshold", DEFAULT_THRESHOLD))
            logger.info(f"Loaded optimal decision threshold: {state['threshold']}")
    else:
        state["threshold"] = DEFAULT_THRESHOLD
        logger.warning(f"Threshold file not found. Defaulting to {DEFAULT_THRESHOLD}")

    # Load explainers for all loaded pipelines
    try:
        state["explainer"] = ChurnExplainer(PIPELINE_PATH)
        state["explainers"]["logistic_regression"] = state["explainer"]
        logger.info("Initialized default ChurnExplainer for Logistic Regression.")
    except Exception as exc:
        logger.warning(f"Could not pre-initialize default ChurnExplainer ({exc}).")

    for m_id in ["random_forest", "xgboost"]:
        meta = MODEL_REGISTRY.get(m_id)
        if meta and meta["file"].exists():
            try:
                state["explainers"][m_id] = ChurnExplainer(meta["file"])
                logger.info(f"Initialized ChurnExplainer for {m_id}.")
            except Exception as e:
                logger.warning(f"Could not initialize explainer for {m_id}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline, threshold, and SHAP explainer on startup."""
    load_model_artifacts()
    yield
    logger.info("Shutting down ChurnSense service.")


app = FastAPI(
    title="ChurnSense API",
    description="Customer Churn Prediction and Explainability Engine",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerInput(BaseModel):
    """Complete schema for all 19 raw customer attributes."""
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    OnlineBackup: Literal["No internet service", "No", "Yes"]
    DeviceProtection: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]
    StreamingTV: Literal["No internet service", "No", "Yes"]
    StreamingMovies: Literal["No internet service", "No", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0.0, le=300.0, description="Monthly fee in USD")
    TotalCharges: float = Field(..., ge=0.0, le=15000.0, description="Total billed amount in USD")


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: Literal["Will churn", "Will stay"]
    threshold: float
    model_used: Optional[str] = "logistic_regression"
    model_name: Optional[str] = "Logistic Regression"
    consensus: Optional[Dict[str, float]] = None


class FeatureReason(BaseModel):
    feature: str
    value: str
    impact: float
    direction: Literal["increases churn risk", "decreases churn risk"]


class ExplainResponse(BaseModel):
    churn_probability: float
    prediction: Literal["Will churn", "Will stay"]
    threshold: float
    top_reasons: List[FeatureReason]
    explanation: str
    explanation_source: Literal["gemini", "fallback"]
    model_used: Optional[str] = "logistic_regression"
    model_name: Optional[str] = "Logistic Regression"
    consensus: Optional[Dict[str, float]] = None


class ChatMessage(BaseModel):
    role: Literal["user", "model", "assistant"]
    text: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: Optional[List[ChatMessage]] = None
    customer_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    source: Literal["gemini", "fallback"]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic 422 errors into friendly field-by-field messages."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", []) if l != "body"])
        errors.append({
            "field": loc or "input",
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "validation_error"),
        })
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid customer input data", "details": errors},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all 500 error handler with clean logging."""
    logger.error(f"Internal server error processing {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error occurred", "message": str(exc)},
    )


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint confirming pipeline readiness."""
    load_model_artifacts()
    pipeline_ready = state["pipeline"] is not None
    return {
        "status": "ok" if pipeline_ready else "initializing",
        "service": "ChurnSense",
    }


@app.get("/models")
def get_available_models():
    """List available model architectures and their benchmark metadata."""
    load_model_artifacts()
    models_list = []
    for m_id, meta in MODEL_REGISTRY.items():
        models_list.append({
            "id": meta["id"],
            "name": meta["name"],
            "subtitle": meta["subtitle"],
            "cv_roc_auc": meta["cv_roc_auc"],
            "test_roc_auc": meta["test_roc_auc"],
            "test_recall": meta["test_recall"],
            "test_precision": meta["test_precision"],
            "test_f1": meta["test_f1"],
            "status": meta["status"],
            "badge": meta["badge"],
            "description": meta["description"],
            "loaded": m_id in state["pipelines"],
        })
    return models_list


@app.get("/demo-cases")
def get_demo_cases():
    """Return the 3 canonical benchmark demo cases with their precomputed metrics."""
    return [
        {
            "id": "high-risk",
            "title": "Case 1: High Risk Persona",
            "subtitle": "Month-to-Month • Fiber Optic • 2 Mo Tenure",
            "expected_risk": "88.9%",
            "verdict": "Will churn",
            "badge_color": "red",
            "description": "Brand-new customer paying high fiber optic charges without retention locks or support.",
            "data": {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.50,
                "TotalCharges": 191.00,
            },
        },
        {
            "id": "low-risk",
            "title": "Case 2: Loyal Persona",
            "subtitle": "2-Year Contract • DSL • 65 Mo Tenure",
            "expected_risk": "7.0%",
            "verdict": "Will stay",
            "badge_color": "green",
            "description": "Long-term subscriber with bundled security, tech support, and automatic card billing.",
            "data": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 65,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 45.00,
                "TotalCharges": 2925.00,
            },
        },
        {
            "id": "borderline",
            "title": "Case 3: Borderline Persona",
            "subtitle": "1-Year Contract • Fiber Optic • 16 Mo Tenure",
            "expected_risk": "57.3%",
            "verdict": "Will churn",
            "badge_color": "amber",
            "description": "Moderate tenure on 1-year contract, hovering near the 0.50 cutoff due to high fiber cost.",
            "data": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 16,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 75.00,
                "TotalCharges": 1200.00,
            },
        },
    ]


def _score_customer(
    customer: CustomerInput, model_key: str = "logistic_regression"
) -> Tuple[float, str, float, str, str, Dict[str, float]]:
    """Helper to score CustomerInput with selected model and compute consensus."""
    load_model_artifacts()

    norm_key = model_key.lower().strip().replace("-", "_").replace(" ", "_")
    if norm_key not in state["pipelines"]:
        norm_key = "logistic_regression"

    pipeline = state["pipelines"].get(norm_key, state["pipeline"])
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model pipeline '{norm_key}' not loaded",
        )

    threshold = state["threshold"]
    input_df = pd.DataFrame([customer.model_dump()])
    proba = float(pipeline.predict_proba(input_df)[0, 1])
    verdict = "Will churn" if proba >= threshold else "Will stay"

    # Multi-model consensus computation
    consensus = {}
    for m_id, pipe in state["pipelines"].items():
        try:
            consensus[m_id] = round(float(pipe.predict_proba(input_df)[0, 1]), 4)
        except Exception:
            pass

    display_name = MODEL_REGISTRY.get(norm_key, {}).get("name", "Logistic Regression")
    return round(proba, 4), verdict, threshold, norm_key, display_name, consensus


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput, model: str = "logistic_regression"):
    """Predict customer churn probability and verdict based on trained optimal threshold."""
    proba, verdict, threshold, model_used, model_name, consensus = _score_customer(customer, model_key=model)
    return {
        "churn_probability": proba,
        "prediction": verdict,
        "threshold": threshold,
        "model_used": model_used,
        "model_name": model_name,
        "consensus": consensus,
    }


@app.post("/explain", response_model=ExplainResponse)
def explain_churn(customer: CustomerInput, model: str = "logistic_regression"):
    """Score customer, compute SHAP feature importance, and generate Gemini/fallback explanation."""
    proba, verdict, threshold, model_used, model_name, consensus = _score_customer(customer, model_key=model)

    explainer = state["explainers"].get(model_used, state.get("explainer"))
    if explainer is None:
        meta = MODEL_REGISTRY.get(model_used)
        path = meta["file"] if meta and meta["file"].exists() else PIPELINE_PATH
        explainer = ChurnExplainer(path)
        state["explainers"][model_used] = explainer

    customer_dict = customer.model_dump()
    top_reasons = explainer.get_top_features(customer_dict, k=3)

    # Gemini LLM explanation with fallback
    explanation_text, explanation_source = explain(verdict, proba, top_reasons)

    return {
        "churn_probability": proba,
        "prediction": verdict,
        "threshold": threshold,
        "top_reasons": top_reasons,
        "explanation": explanation_text,
        "explanation_source": explanation_source,
        "model_used": model_used,
        "model_name": model_name,
        "consensus": consensus,
    }


@app.post("/chat", response_model=ChatResponse)
def chat_with_advisor(req: ChatRequest):
    """Interactive AI retention advisor endpoint for answering user doubts and retention queries."""
    history_dicts = [h.model_dump() for h in req.history] if req.history else None
    reply, source = ask_churn_assistant(
        message=req.message,
        history=history_dicts,
        customer_context=req.customer_context,
    )
    return {
        "reply": reply,
        "source": source,
    }


# Static assets & index route
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page application frontend."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend index.html is being prepared."},
        )
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
