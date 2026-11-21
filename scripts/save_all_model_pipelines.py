"""Fit and save all three candidate pipelines (Logistic Regression, Random Forest, XGBoost)
to allow real-time model switching in ChurnSense.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import DATA_RAW_FILE, RANDOM_STATE, MODELS_DIR, PIPELINE_PATH
from src.preprocess import clean_data
from src.pipeline import build_pipeline


def save_all_pipelines():
    print("Preparing data for all models...")
    raw_df = pd.read_csv(DATA_RAW_FILE)
    X, y = clean_data(raw_df, verbose=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives

    models = {
        "logistic_regression": LogisticRegression(
            C=10.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key, model in models.items():
        print(f"Fitting pipeline for {key}...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        out_path = MODELS_DIR / f"pipeline_{key}.joblib"
        joblib.dump(pipeline, out_path)
        print(f"  -> Saved {out_path}")

        # If logistic regression, ensure PIPELINE_PATH is also kept in sync
        if key == "logistic_regression":
            joblib.dump(pipeline, PIPELINE_PATH)
            print(f"  -> Synchronized production pipeline at {PIPELINE_PATH}")

    print("All model pipelines saved successfully!")


if __name__ == "__main__":
    save_all_pipelines()
