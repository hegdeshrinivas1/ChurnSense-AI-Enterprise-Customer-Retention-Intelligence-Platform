"""Manual evaluation metrics and comparison against scikit-learn."""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute True Positives, False Positives, False Negatives, True Negatives manually.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Tuple of (tp, fp, fn, tn) counts.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    return tp, fp, fn, tn


def manual_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate accuracy from raw confusion matrix counts."""
    total = tp + fp + fn + tn
    return float((tp + tn) / total) if total > 0 else 0.0


def manual_precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate precision (positive predictive value) from raw counts."""
    denominator = tp + fp
    return float(tp / denominator) if denominator > 0 else 0.0


def manual_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate recall (sensitivity / true positive rate) from raw counts."""
    denominator = tp + fn
    return float(tp / denominator) if denominator > 0 else 0.0


def manual_f1(tp: int, fp: int, fn: int, tn: int) -> float:
    """Calculate F1-score as harmonic mean of precision and recall."""
    prec = manual_precision(tp, fp, fn, tn)
    rec = manual_recall(tp, fp, fn, tn)
    return float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0


def compute_all_manual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute dictionary of all manual metrics from binary labels."""
    tp, fp, fn, tn = compute_confusion_counts(y_true, y_pred)
    return {
        "accuracy": manual_accuracy(tp, fp, fn, tn),
        "precision": manual_precision(tp, fp, fn, tn),
        "recall": manual_recall(tp, fp, fn, tn),
        "f1": manual_f1(tp, fp, fn, tn),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def compare_metrics(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True) -> pd.DataFrame:
    """Compare manual metric calculations against scikit-learn implementations.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        verbose: Whether to print formatted comparison table.

    Returns:
        DataFrame detailing manual vs sklearn metrics and absolute difference.
    """
    tp, fp, fn, tn = compute_confusion_counts(y_true, y_pred)

    m_acc = manual_accuracy(tp, fp, fn, tn)
    m_prec = manual_precision(tp, fp, fn, tn)
    m_rec = manual_recall(tp, fp, fn, tn)
    m_f1 = manual_f1(tp, fp, fn, tn)

    sk_acc = float(accuracy_score(y_true, y_pred))
    sk_prec = float(precision_score(y_true, y_pred, zero_division=0))
    sk_rec = float(recall_score(y_true, y_pred, zero_division=0))
    sk_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    data = [
        {"Metric": "Accuracy", "Manual": m_acc, "Scikit-Learn": sk_acc, "Abs Diff": abs(m_acc - sk_acc)},
        {"Metric": "Precision", "Manual": m_prec, "Scikit-Learn": sk_prec, "Abs Diff": abs(m_prec - sk_prec)},
        {"Metric": "Recall", "Manual": m_rec, "Scikit-Learn": sk_rec, "Abs Diff": abs(m_rec - sk_rec)},
        {"Metric": "F1-Score", "Manual": m_f1, "Scikit-Learn": sk_f1, "Abs Diff": abs(m_f1 - sk_f1)},
    ]

    df_comp = pd.DataFrame(data)

    if verbose:
        print("\n" + "=" * 55)
        print("METRIC VERIFICATION: MANUAL VS SCIKIT-LEARN")
        print("=" * 55)
        print(f"Confusion counts: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(df_comp.to_string(index=False, formatters={
            "Manual": "{:.5f}".format,
            "Scikit-Learn": "{:.5f}".format,
            "Abs Diff": "{:.2e}".format,
        }))
        print("=" * 55 + "\n")

    # Assert matching within numerical tolerance
    for item in data:
        assert item["Abs Diff"] < 1e-9, f"Mismatch in {item['Metric']}: {item['Manual']} != {item['Scikit-Learn']}"

    return df_comp


if __name__ == "__main__":
    # Test on synthetic binary data
    test_y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    test_y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    compare_metrics(test_y_true, test_y_pred, verbose=True)
