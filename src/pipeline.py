"""Scikit-Learn Pipeline definition for ChurnSense."""

from typing import Any, List
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def get_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer for numeric scaling and categorical one-hot encoding."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Build unified end-to-end Pipeline taking raw tabular fields directly to predictions.

    Args:
        model: Sklearn-compatible classifier instance.

    Returns:
        Fitted or unfitted Pipeline containing preprocessor and classifier.
    """
    preprocessor = get_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract output feature names from the fitted preprocessor inside the pipeline."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from src.preprocess import clean_data
    from src.config import DATA_RAW_FILE

    df = pd.read_csv(DATA_RAW_FILE)
    X, y = clean_data(df, verbose=False)
    pipe = build_pipeline(LogisticRegression())
    pipe.fit(X.iloc[:100], y.iloc[:100])
    feature_names = get_feature_names(pipe)
    print(f"Pipeline successfully built and tested!")
    print(f"Total encoded features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:8]}")
