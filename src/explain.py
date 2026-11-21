"""SHAP explainability module for ChurnSense."""

from typing import List, Dict, Any, Union
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

from src.config import PIPELINE_PATH, SHAP_SUMMARY_PATH, DATA_RAW_FILE, RANDOM_STATE
from src.preprocess import clean_data
from src.pipeline import get_feature_names


def format_feature_name(raw_name: str) -> str:
    """Transform encoded pipeline feature name into clear, human-readable name."""
    name = raw_name
    if name.startswith("num__"):
        col = name.replace("num__", "")
        mapping = {
            "tenure": "Tenure (months)",
            "MonthlyCharges": "Monthly Charges ($)",
            "TotalCharges": "Total Charges ($)",
        }
        return mapping.get(col, col)

    if name.startswith("cat__"):
        clean_cat = name.replace("cat__", "")
        # e.g., Contract_Month-to-month -> Contract: Month-to-month
        if "_" in clean_cat:
            parts = clean_cat.split("_", 1)
            prefix = parts[0]
            val = parts[1]
            return f"{prefix}: {val}"
        return clean_cat

    return name


class ChurnExplainer:
    """Manages SHAP model explanations for both individual inference and global summaries."""

    def __init__(self, pipeline_path: Path = PIPELINE_PATH):
        """Load pipeline, extract preprocessor and classifier, initialize SHAP explainer."""
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Trained pipeline not found at {pipeline_path}. Run train.py first.")

        self.pipeline = joblib.load(pipeline_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.readable_feature_names = [format_feature_name(f) for f in self.feature_names]

        # Load reference background dataset for SHAP
        raw_df = pd.read_csv(DATA_RAW_FILE)
        X, _ = clean_data(raw_df, verbose=False)
        self.background_raw = X.sample(n=min(200, len(X)), random_state=RANDOM_STATE)
        self.background_transformed = self.preprocessor.transform(self.background_raw)

        # Initialize appropriate explainer
        cls_name = self.classifier.__class__.__name__
        if "Forest" in cls_name or "XGB" in cls_name or "Tree" in cls_name:
            self.explainer = shap.TreeExplainer(self.classifier)
        elif "Logistic" in cls_name:
            # LinearExplainer works directly on transformed linear inputs
            self.explainer = shap.LinearExplainer(self.classifier, self.background_transformed)
        else:
            self.explainer = shap.Explainer(self.classifier, self.background_transformed)

    def explain_instance(self, input_df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a single transformed instance or small batch."""
        X_trans = self.preprocessor.transform(input_df)
        shap_values = self.explainer(X_trans)

        # Extract values for positive class (Churn = 1)
        values = shap_values.values
        if values.ndim == 3:  # (samples, features, classes)
            return values[:, :, 1]
        return values

    def get_top_features(
        self, input_row: Union[pd.DataFrame, pd.Series, Dict[str, Any]], k: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract the top-k most influential features for an input record.

        Args:
            input_row: Raw single-row DataFrame, Series, or dictionary of customer fields.
            k: Number of top features to return.

        Returns:
            List of dicts with keys: feature, value, impact, direction.
        """
        if isinstance(input_row, pd.Series):
            df_instance = input_row.to_frame().T
        elif isinstance(input_row, dict):
            df_instance = pd.DataFrame([input_row])
        else:
            df_instance = input_row.copy()

        # Compute SHAP array for this instance
        shap_vals = self.explain_instance(df_instance)[0]

        # Rank features by absolute impact
        abs_indices = np.argsort(np.abs(shap_vals))[::-1]

        top_features = []
        for idx in abs_indices[:k]:
            raw_encoded_name = self.feature_names[idx]
            clean_name = self.readable_feature_names[idx]
            impact = float(shap_vals[idx])
            direction = "increases churn risk" if impact > 0 else "decreases churn risk"

            # Determine corresponding raw feature value
            if raw_encoded_name.startswith("num__"):
                orig_col = raw_encoded_name.replace("num__", "")
                val = df_instance[orig_col].iloc[0] if orig_col in df_instance.columns else "N/A"
            elif raw_encoded_name.startswith("cat__"):
                clean_cat = raw_encoded_name.replace("cat__", "")
                parts = clean_cat.split("_", 1)
                orig_col = parts[0]
                val = df_instance[orig_col].iloc[0] if orig_col in df_instance.columns else parts[1]
            else:
                val = "N/A"

            top_features.append({
                "feature": clean_name,
                "value": str(val),
                "impact": round(impact, 4),
                "direction": direction,
            })

        return top_features

    def generate_global_summary_plot(self, output_path: Path = SHAP_SUMMARY_PATH):
        """Generate and save global SHAP beeswarm / bar summary plot."""
        print(f"Generating global SHAP summary plot at {output_path}...")
        shap_vals = self.explainer(self.background_transformed)
        values = shap_vals.values
        if values.ndim == 3:
            values = values[:, :, 1]

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            values,
            self.background_transformed,
            feature_names=self.readable_feature_names,
            max_display=12,
            show=False,
        )
        plt.title("Global Feature Importance (SHAP Summary)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Saved global SHAP plot to {output_path}")


if __name__ == "__main__":
    explainer = ChurnExplainer()
    explainer.generate_global_summary_plot()

    # Test top features on sample customer
    sample_df = pd.read_csv(DATA_RAW_FILE)
    X, _ = clean_data(sample_df, verbose=False)
    sample_cust = X.iloc[0]
    top_3 = explainer.get_top_features(sample_cust, k=3)
    print("\nSample Customer Top 3 Reasons:")
    for f in top_3:
        print(f" - {f['feature']} (value: {f['value']}): impact {f['impact']} -> {f['direction']}")
