"""LLM plain-language explanation generator with robust fallback handling."""

from typing import List, Dict, Any, Tuple
import os
import logging
from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("churnsense.llm")


def generate_fallback_explanation(
    prediction: str, probability: float, top_features: List[Dict[str, Any]]
) -> str:
    """Generate deterministic template-based explanation when Gemini API is unavailable.

    Args:
        prediction: "Will churn" or "Will stay".
        probability: Float churn probability (0.0 to 1.0).
        top_features: List of top contributing features with feature, value, impact, direction.

    Returns:
        Two simple sentences describing the prediction drivers.
    """
    prob_pct = round(probability * 100, 1)

    if not top_features:
        return (
            f"The customer is predicted to {prediction.lower()} with a {prob_pct}% probability. "
            f"Overall tenure and contract characteristics are the primary indicators for this assessment."
        )

    f1 = top_features[0]
    direction_verb1 = "increases churn risk" if f1["impact"] > 0 else "reduces churn risk"

    if len(top_features) > 1:
        f2 = top_features[1]
        direction_verb2 = "elevating risk" if f2["impact"] > 0 else "supporting retention"
        sentence1 = (
            f"This customer is predicted to {prediction.lower()} with a {prob_pct}% churn probability, "
            f"heavily influenced by {f1['feature']} ({f1['value']}) which {direction_verb1}."
        )
        sentence2 = (
            f"Additionally, {f2['feature']} ({f2['value']}) plays a secondary role by {direction_verb2}."
        )
        return f"{sentence1} {sentence2}"

    return (
        f"This customer is predicted to {prediction.lower()} with a {prob_pct}% churn probability. "
        f"The primary driver is {f1['feature']} ({f1['value']}), which {direction_verb1}."
    )


def explain(
    prediction: str, probability: float, top_features: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Generate a 2-sentence plain language explanation via Gemini or fallback template.

    Args:
        prediction: Verdict string ("Will churn" or "Will stay").
        probability: Churn probability score.
        top_features: Top SHAP features.

    Returns:
        Tuple of (explanation_text, explanation_source) where source is 'gemini' or 'fallback'.
    """
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.info("No Gemini API key detected. Using fallback explanation.")
        return generate_fallback_explanation(prediction, probability, top_features), "fallback"

    # Format features for the prompt
    features_lines = []
    for f in top_features:
        features_lines.append(
            f"- {f['feature']} (observed value: {f['value']}) -> {f['direction']} (SHAP impact: {f['impact']:+.3f})"
        )
    features_text = "\n".join(features_lines)
    prob_pct = round(probability * 100, 1)

    prompt = (
        f"You are a professional customer retention analyst.\n\n"
        f"Customer Status: {prediction}\n"
        f"Predicted Churn Probability: {prob_pct}%\n"
        f"Top 3 Model Drivers (SHAP values):\n{features_text}\n\n"
        f"Explain in 2 simple sentences why this customer is likely to leave or stay. "
        f"Use ONLY the features provided. Do not invent reasons, numbers or offers."
    )

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            cleaned_text = response.text.strip().replace("\n", " ")
            # Ensure no excess spaces
            cleaned_text = " ".join(cleaned_text.split())
            return cleaned_text, "gemini"
        else:
            logger.warning("Empty response from Gemini API, falling back.")
            return generate_fallback_explanation(prediction, probability, top_features), "fallback"

    except Exception as exc:
        logger.warning(f"Gemini API invocation failed ({exc}). Gracefully using fallback explanation.")
        return generate_fallback_explanation(prediction, probability, top_features), "fallback"


if __name__ == "__main__":
    sample_features = [
        {"feature": "Monthly Charges ($)", "value": "105.50", "impact": 1.85, "direction": "increases churn risk"},
        {"feature": "Contract: Month-to-month", "value": "Month-to-month", "impact": 1.42, "direction": "increases churn risk"},
        {"feature": "Tenure (months)", "value": "3", "impact": 0.95, "direction": "increases churn risk"},
    ]

    exp_text, source = explain("Will churn", 0.824, sample_features)
    print(f"Explanation Source: {source}")
    print(f"Explanation:\n{exp_text}")
