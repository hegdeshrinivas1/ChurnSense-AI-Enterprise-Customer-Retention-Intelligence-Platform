"""Model training, comparison, MLflow tracking, hyperparameter tuning, and threshold selection."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import (
    DATA_RAW_FILE,
    RANDOM_STATE,
    MLFLOW_EXPERIMENT_NAME,
    MLRUNS_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PIPELINE_PATH,
    THRESHOLD_PATH,
)
from src.preprocess import clean_data
from src.pipeline import build_pipeline
from src.metrics import compute_all_manual_metrics, compare_metrics


def evaluate_predictions_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Calculate metrics at a specific decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_all_manual_metrics(y_true, y_pred)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = 0.0
    metrics["threshold"] = threshold
    return metrics


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained (0)", "Churn (1)"],
                yticklabels=["Retained (0)", "Churn (1)"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def train_and_compare_models():
    """Execute stratified train/test split, 5-fold CV, MLflow tracking, and model comparison."""
    print("Loading and preparing data...")
    raw_df = pd.read_csv(DATA_RAW_FILE)
    X, y = clean_data(raw_df, verbose=False)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives

    print(f"Training set: {X_train.shape[0]} samples (Churn: {positives}, Retained: {negatives})")
    print(f"Test set:     {X_test.shape[0]} samples (Churn: {(y_test == 1).sum()}, Retained: {(y_test == 0).sum()})")

    # MLflow Setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    candidate_models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_records = []
    fitted_pipelines = {}

    print("\nEvaluating candidate models with 5-fold Stratified CV & Test Evaluation...")
    for model_name, model in candidate_models.items():
        pipeline = build_pipeline(model)

        # 5-fold CV on train set
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=["roc_auc", "f1"],
            n_jobs=-1,
            return_train_score=False,
        )

        cv_roc_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
        cv_roc_auc_std = float(np.std(cv_results["test_roc_auc"]))
        cv_f1_mean = float(np.mean(cv_results["test_f1"]))
        cv_f1_std = float(np.std(cv_results["test_f1"]))

        # Fit on full train set and evaluate on test set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions_at_threshold(y_test.values, y_test_proba, threshold=0.50)

        # Log run in MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_roc_auc_mean", cv_roc_auc_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_roc_auc_std)
            mlflow.log_metric("cv_f1_mean", cv_f1_mean)
            mlflow.log_metric("cv_f1_std", cv_f1_std)
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
            mlflow.log_metric("test_precision", test_metrics["precision"])
            mlflow.log_metric("test_recall", test_metrics["recall"])
            mlflow.log_metric("test_f1", test_metrics["f1"])
            mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])

            # Confusion matrix artifact
            cm_img_path = MODELS_DIR / f"cm_{model_name.lower()}.png"
            y_pred_default = (y_test_proba >= 0.50).astype(int)
            plot_and_save_confusion_matrix(
                y_test.values, y_pred_default, f"{model_name} Test Confusion Matrix", cm_img_path
            )
            mlflow.log_artifact(str(cm_img_path))

            # Model artifact
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

        comparison_records.append({
            "Model": model_name,
            "CV ROC-AUC (mean±std)": f"{cv_roc_auc_mean:.4f} ± {cv_roc_auc_std:.4f}",
            "CV F1 (mean±std)": f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}",
            "Test Precision": test_metrics["precision"],
            "Test Recall": test_metrics["recall"],
            "Test F1": test_metrics["f1"],
            "Test ROC-AUC": test_metrics["roc_auc"],
            "_cv_auc_num": cv_roc_auc_mean,
            "_test_f1_num": test_metrics["f1"],
        })

    df_comp = pd.DataFrame(comparison_records)
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE (Real Measured Numbers)")
    print("=" * 90)
    display_cols = [
        "Model",
        "CV ROC-AUC (mean±std)",
        "CV F1 (mean±std)",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Test ROC-AUC",
    ]
    print(df_comp[display_cols].to_string(index=False, formatters={
        "Test Precision": "{:.4f}".format,
        "Test Recall": "{:.4f}".format,
        "Test F1": "{:.4f}".format,
        "Test ROC-AUC": "{:.4f}".format,
    }))
    print("=" * 90)

    # Pick best model based on CV ROC-AUC
    best_row = df_comp.sort_values(by="_cv_auc_num", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\nBest baseline candidate model: {best_model_name} (CV ROC-AUC: {best_row['_cv_auc_num']:.4f})")

    # PHASE 8: Tuning & Threshold Selection
    print("\n" + "=" * 70)
    print(f"TUNING & THRESHOLD OPTIMIZATION FOR {best_model_name.upper()}")
    print("=" * 70)

    tuned_pipeline = tune_model(best_model_name, X_train, y_train, scale_pos_weight)

    # Evaluate tuned model on test set
    y_test_proba_tuned = tuned_pipeline.predict_proba(X_test)[:, 1]
    tuned_auc = float(roc_auc_score(y_test, y_test_proba_tuned))
    print(f"Tuned {best_model_name} Test ROC-AUC: {tuned_auc:.4f}")

    # Threshold scan from 0.20 to 0.70 (step 0.05)
    thresholds = [round(t, 2) for t in np.arange(0.20, 0.71, 0.05)]
    threshold_records = []
    for t in thresholds:
        m = evaluate_predictions_at_threshold(y_test.values, y_test_proba_tuned, t)
        threshold_records.append({
            "Threshold": t,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1"],
            "Accuracy": m["accuracy"],
            "TP": int(m["tp"]),
            "FP": int(m["fp"]),
            "FN": int(m["fn"]),
            "TN": int(m["tn"]),
        })

    df_thresh = pd.DataFrame(threshold_records)
    print("\nDecision Threshold Scan (0.20 to 0.70):")
    print(df_thresh[["Threshold", "Precision", "Recall", "F1-Score", "Accuracy", "TP", "FP", "FN", "TN"]].to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format,
            "Accuracy": "{:.4f}".format,
        }
    ))

    # Selection rule: recall >= 0.75 with best precision. If none >= 0.75, best recall.
    eligible = df_thresh[df_thresh["Recall"] >= 0.75]
    if not eligible.empty:
        best_thresh_row = eligible.sort_values(by="Precision", ascending=False).iloc[0]
        reason = (
            f"Selected threshold {best_thresh_row['Threshold']:.2f} satisfies the operational requirement "
            f"of Recall >= 0.75 ({best_thresh_row['Recall']:.4f}) while maximizing Precision ({best_thresh_row['Precision']:.4f}) "
            f"and achieving an F1-score of {best_thresh_row['F1-Score']:.4f}."
        )
    else:
        best_thresh_row = df_thresh.sort_values(by="Recall", ascending=False).iloc[0]
        reason = (
            f"No threshold achieved Recall >= 0.75; selected threshold {best_thresh_row['Threshold']:.2f} "
            f"which yielded the highest available Recall of {best_thresh_row['Recall']:.4f} with Precision {best_thresh_row['Precision']:.4f}."
        )

    chosen_threshold = float(best_thresh_row["Threshold"])
    print(f"\n{reason}")

    # Verify manual metrics against scikit-learn on final chosen threshold
    y_pred_chosen = (y_test_proba_tuned >= chosen_threshold).astype(int)
    compare_metrics(y_test.values, y_pred_chosen, verbose=True)

    # Save threshold.json
    threshold_data = {
        "model_name": best_model_name,
        "optimal_threshold": chosen_threshold,
        "test_roc_auc": round(tuned_auc, 4),
        "test_precision": round(float(best_thresh_row["Precision"]), 4),
        "test_recall": round(float(best_thresh_row["Recall"]), 4),
        "test_f1": round(float(best_thresh_row["F1-Score"]), 4),
        "test_accuracy": round(float(best_thresh_row["Accuracy"]), 4),
        "explanation": reason,
        "confusion_matrix": {
            "tp": int(best_thresh_row["TP"]),
            "fp": int(best_thresh_row["FP"]),
            "fn": int(best_thresh_row["FN"]),
            "tn": int(best_thresh_row["TN"]),
        }
    }

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved threshold configuration to {THRESHOLD_PATH}")

    # Save final pipeline
    joblib.dump(tuned_pipeline, PIPELINE_PATH)
    print(f"Saved optimal production pipeline to {PIPELINE_PATH}")

    # Log best tuned model in MLflow
    with mlflow.start_run(run_name=f"{best_model_name}_Tuned_Optimal"):
        mlflow.log_param("model_type", f"{best_model_name}_Tuned")
        mlflow.log_param("optimal_threshold", chosen_threshold)
        mlflow.log_metric("test_roc_auc", tuned_auc)
        mlflow.log_metric("optimal_precision", float(best_thresh_row["Precision"]))
        mlflow.log_metric("optimal_recall", float(best_thresh_row["Recall"]))
        mlflow.log_metric("optimal_f1", float(best_thresh_row["F1-Score"]))
        mlflow.log_artifact(str(THRESHOLD_PATH))
        mlflow.sklearn.log_model(
            tuned_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    return df_comp, threshold_data


def tune_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
    """Run RandomizedSearchCV on chosen model type."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if model_name == "LogisticRegression":
        base_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        param_dist = {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "saga"],
        }
    elif model_name == "RandomForest":
        base_model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__max_depth": [4, 6, 8, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    else:  # XGBoost
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        param_dist = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        }

    pipeline = build_pipeline(base_model)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    return search.best_estimator_


if __name__ == "__main__":
    train_and_compare_models()
