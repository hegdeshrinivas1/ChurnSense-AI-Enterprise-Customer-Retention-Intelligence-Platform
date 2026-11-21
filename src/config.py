"""Configuration settings, paths, and constants for ChurnSense."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_FILE = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary.png"
DOCS_SCREENSHOTS_DIR = BASE_DIR / "docs" / "screenshots"
WEB_DIR = BASE_DIR / "web"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset constants
TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Model & training parameters
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.40
MLFLOW_EXPERIMENT_NAME = "churnsense"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PORT = int(os.getenv("PORT", 8000))
