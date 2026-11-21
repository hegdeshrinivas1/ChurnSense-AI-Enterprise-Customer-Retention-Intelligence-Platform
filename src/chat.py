"""Chatbot assistant engine for ChurnSense with Gemini integration and comprehensive domain fallback."""

from typing import List, Dict, Any, Optional, Tuple
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("churnsense.chat")

SYSTEM_PROMPT = """You are the ChurnSense AI Retention Advisor, an expert assistant embedded in the ChurnSense customer churn prediction platform.
Your expertise covers:
1. Platform navigation & usage: explain how to use the predictor, interpret the circular gauge, understand SHAP driver cards, and test presets.
2. Telecom customer churn mechanics: contract influence, fiber optic pricing, bundling value-added services (tech support, online security).
3. Machine learning interpretability: ROC-AUC (0.8408), recall (78.34%), calibrated decision threshold (0.50), and SHAP feature attributions.
4. Actionable retention interventions: customer save actions, contract upgrade discounts, loyalty credits, and support trials.
5. Analyzing the specific customer currently active in the user's session if customer context is provided.

Guidelines:
- If asked "how to use" or general questions, provide a clear, structured step-by-step answer.
- Keep answers concise, direct, helpful, and professional (2-3 short paragraphs or bullet points).
- If customer context is provided, reference their actual numbers (e.g. tenure, monthly charges, contract) to ground your advice.
- Maintain a friendly, supportive SaaS advisor tone.
"""


def generate_fallback_chat_reply(
    message: str, customer_context: Optional[Dict[str, Any]] = None
) -> str:
    """Generate intelligent, context-aware responses when Gemini API is unavailable or times out."""
    msg = message.lower().strip()

    # 1. Platform Navigation & Usage Queries
    usage_keywords = [
        "how to use", "how do i use", "how can i use", "how does this work",
        "how does it work", "how do this work", "how does the page work",
        "how does the website work", "how to start", "get started", "getting started",
        "how do i start", "what to do", "what should i do", "guide", "tutorial",
        "walkthrough", "help me use", "usage", "instructions", "how to test",
        "how do i test", "steps to use", "step by step"
    ]
    if any(k in msg for k in usage_keywords) or msg in ["help", "how to", "usage", "guide"]:
        return (
            "Here is how to use the **ChurnSense Platform** in 4 simple steps:\n\n"
            "1. **Select or Configure Customer Attributes** (Left Panel):\n"
            "   • Click one of the quick persona presets (**⚡ High Risk**, **🛡️ Low Risk**, or **⚖️ Borderline**) to autofill realistic customer data, or manually adjust demographic, service, and billing sliders.\n\n"
            "2. **Click 'Predict & Explain Churn'**:\n"
            "   • The unified Scikit-Learn pipeline instantly scales numeric features, encodes categories, and evaluates churn probability in <15ms.\n\n"
            "3. **Inspect the Risk Intelligence Panel** (Right Panel):\n"
            "   • Check the **Animated Circular Gauge** for the exact churn probability and calibrated risk verdict.\n"
            "   • Review the **Top 3 SHAP Drivers** to see which specific features increase (▲ red) or decrease (▼ green) customer churn risk.\n"
            "   • Read the **AI Synthesis Card** for an executive 2-sentence plain-English summary.\n\n"
            "4. **Ask Me for Retention Strategies**:\n"
            "   • In this chat, click **💡 Retention Offer** or ask me how to save the customer!"
        )

    # 2. About / Platform Overview
    overview_keywords = [
        "what is churnsense", "about this project", "overview", "what does this project do",
        "what this project does", "what does this do", "what this does", "who are you",
        "explain this project", "explain the project", "project details",
        "tell me about this project", "what is this page", "what is this website"
    ]
    if any(k in msg for k in overview_keywords):
        return (
            "**ChurnSense** is an enterprise-grade customer churn intelligence system built on the IBM Telco Customer Churn dataset.\n\n"
            "• **Prediction Engine**: Evaluates customer churn probability using a tuned classification pipeline with an **84.08% Test ROC-AUC**.\n"
            "• **Cost-Sensitive Decision Threshold**: Calibrated at **0.50** to capture **78.34% of churners** (4 out of 5 at-risk customers).\n"
            "• **Explainable AI (XAI)**: Uses SHAP game theory to isolate the top factors driving each individual customer's decision.\n"
            "• **Generative AI Synthesis**: Integrates Google's Gemini Flash model to translate ML scores into executive retention actions."
        )

    # 3. Context-Aware Queries (Active Customer in Session)
    if customer_context:
        prob = customer_context.get("churn_probability")
        verdict = customer_context.get("prediction", "evaluated")
        reasons = customer_context.get("top_reasons", [])
        reasons_str = ", ".join([f"{r.get('feature')} ({r.get('value')})" for r in reasons[:2]]) if reasons else "contract and billing structure"
        prob_pct = f"{round(prob * 100, 1)}%" if prob is not None else "elevated"

        # Why is this customer at risk / churn reasons
        if any(w in msg for w in ["why", "reason", "risk", "leave", "churn", "leaving"]):
            return (
                f"**Churn Risk Assessment for Current Customer ({prob_pct} - {verdict}):**\n\n"
                f"According to mathematical SHAP feature attribution, the primary drivers are:\n"
                f"• **Primary Driver**: {reasons[0].get('feature', 'Billing')} ({reasons[0].get('value', 'N/A')}) — {reasons[0].get('direction', 'increases risk')}.\n"
                + (f"• **Secondary Driver**: {reasons[1].get('feature', 'Contract')} ({reasons[1].get('value', 'N/A')}) — {reasons[1].get('direction', 'increases risk')}.\n\n" if len(reasons) > 1 else "\n\n")
                + "In subscription telecom, customers on month-to-month contracts with high monthly charges ($70+) face zero friction to switch to promotional competitor rates unless secured with long-term retention anchors."
            )

        # Retention offer / save recommendation
        if any(w in msg for w in ["offer", "retain", "save", "action", "strategy", "recommend", "discount"]):
            return (
                f"**Targeted Retention Playbook for Current Customer ({verdict}):**\n\n"
                f"1. **Contract Upgrade Incentive**: Offer a **15% monthly bill credit** in exchange for transitioning from month-to-month to a **1-year contract**. In our dataset, annual contracts reduce churn by >70%.\n"
                f"2. **Complimentary Value Bundle**: Add **3 free months of Tech Support & Online Security**. Customers with bundled security services have less than half the attrition rate of unbundled accounts.\n"
                f"3. **Payment Method Modernization**: Provide a one-time $20 statement credit to enroll in **Automatic Bank/Card Payments**, preventing billing-friction churn associated with paper or electronic checks."
            )

    # 4. Explainable AI & SHAP Queries
    if any(w in msg for w in ["shap", "explainability", "attribution", "shapley", "game theory"]):
        return (
            "**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach that explains individual machine learning predictions:\n\n"
            "• **Fair Attribution**: Treats features as 'players' in a game and computes their exact marginal contribution to the final risk score.\n"
            "• **Directional Impact**: Positive values (▲ red bars) push the score toward churn; negative values (▼ green bars) act as retention anchors.\n"
            "• **Local vs Global**: ChurnSense computes local SHAP values for every customer in real time and also aggregates them into a global feature importance plot saved in `models/shap_summary.png`."
        )

    # 5. Threshold & Decision Boundary Queries
    if any(w in msg for w in ["threshold", "decision boundary", "cutoff", "0.50", "0.5"]):
        return (
            "**Why Decision Threshold Calibration Matters:**\n\n"
            "Standard classifiers use an arbitrary 0.50 cutoff. But in telecom churn, the business cost of a **False Negative** (losing a $1,200/year customer) is far higher than a **False Positive** (sending a retention coupon to someone who would have stayed).\n\n"
            "We scanned thresholds from 0.20 to 0.70 with a business constraint: **Recall must be $\\ge 75\%$ while maximizing Precision**.\n"
            "• **Optimal Threshold**: **0.50**\n"
            "• **Achieved Recall**: **78.34%** (captures nearly 4 out of every 5 churners)\n"
            "• **Achieved Precision**: **50.34%**"
        )

    # 6. ML Model & Evaluation Metrics Queries
    if any(w in msg for w in ["model", "algorithm", "accuracy", "roc", "auc", "precision", "recall", "f1", "comparison", "benchmark"]):
        return (
            "**Model Comparison & Benchmark Results:**\n\n"
            "We evaluated three models using 5-fold Stratified Cross-Validation on 5,634 training records and tested on 1,409 held-out records:\n\n"
            "• **Logistic Regression (Tuned Winner)**: **0.8408 Test ROC-AUC**, 0.6130 F1, 78.34% Recall.\n"
            "• **Random Forest**: 0.8226 Test ROC-AUC, 0.5874 F1, 63.37% Recall.\n"
            "• **XGBoost**: 0.8187 Test ROC-AUC, 0.5877 F1, 66.31% Recall.\n\n"
            "*You can switch between any of these 3 models right in the predictor header tabs to compare how tree vs linear architectures evaluate each customer in real time!*"
        )

    # 7. Dataset & Contract Insights
    if any(w in msg for w in ["data", "dataset", "contract", "tenure", "charges", "ibm"]):
        return (
            "**Key Findings from the IBM Telco Dataset (7,043 Customers):**\n\n"
            "1. **Contract Type**: Month-to-month contracts experience a **42.7% churn rate**, compared to **11.3%** for 1-year and **2.8%** for 2-year contracts.\n"
            "2. **Critical Tenure Window**: Peak churn occurs in the first **0 to 12 months**. After month 24, attrition drops significantly.\n"
            "3. **Pricing Sensitivity**: Churn density spikes between **$70 and $105/month**, especially for fiber optic users without security add-ons.\n"
            "4. **Data Hygiene**: Fixed 11 blank whitespace values in `TotalCharges` for new enrollments (`tenure=0`) by imputing 0.0."
        )

    # 8. Friendly Fallback
    return (
        "I am your **ChurnSense AI Advisor**. Here are key things you can ask me:\n\n"
        "• *'How do I use this page?'* — Step-by-step walkthrough of the dashboard.\n"
        "• *'Why is this customer at risk?'* — Detailed explanation of current customer factors.\n"
        "• *'What retention offer should we make?'* — Recommended save strategies.\n"
        "• *'How does SHAP work?'* — Explanation of feature attribution.\n"
        "• *'Why was threshold 0.50 chosen?'* — Explanation of recall optimization."
    )


def _call_gemini_raw(prompt: str, api_key: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=350,
        temperature=0.3,
    )
    res = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    return res.text if res else ""


def ask_churn_assistant(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    customer_context: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 12.0,
) -> Tuple[str, str]:
    """Process user question using direct knowledge rules or Gemini with strict timeout.

    Args:
        message: The user's question or doubt.
        history: Previous chat exchanges [{"role": "user"|"model", "text": "..."}].
        customer_context: Optional dictionary containing currently analyzed prediction, probability, and features.
        timeout_seconds: Maximum seconds to wait before falling back to rule-based response.

    Returns:
        Tuple of (reply_text, reply_source) where source is 'gemini' or 'fallback'.
    """
    msg_clean = message.lower().strip()

    # Instant zero-latency responses for common platform navigation & help queries
    instant_help_keys = [
        "how to use", "how do i use", "how can i use", "how does this work",
        "how does it work", "how do this work", "how does the page work",
        "how does the website work", "how to start", "get started", "getting started",
        "how do i start", "what to do", "what should i do", "guide", "tutorial",
        "walkthrough", "help me use", "usage", "instructions", "how to test",
        "how do i test", "steps to use", "step by step",
        "what is churnsense", "about this project", "overview", "what does this project do",
        "what this project does", "what does this do", "what this does", "who are you",
        "explain this project", "explain the project", "project details",
        "tell me about this project", "what is this page", "what is this website"
    ]
    if any(k in msg_clean for k in instant_help_keys) or msg_clean in ["help", "how to", "usage", "guide"]:
        return generate_fallback_chat_reply(message, customer_context), "fallback"

    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.info("No Gemini key found. Using fallback chat generator.")
        return generate_fallback_chat_reply(message, customer_context), "fallback"

    # Assemble contextual prompt
    context_str = ""
    if customer_context:
        prob = customer_context.get("churn_probability")
        verdict = customer_context.get("prediction")
        reasons = customer_context.get("top_reasons", [])
        reasons_text = "; ".join([f"{r.get('feature')}: {r.get('value')} ({r.get('direction')})" for r in reasons])
        context_str = (
            f"\n\nCURRENT CUSTOMER IN SESSION:\n"
            f"- Churn Probability: {round(prob * 100, 1) if prob is not None else 'N/A'}%\n"
            f"- Verdict: {verdict}\n"
            f"- Key SHAP Drivers: {reasons_text}\n"
            f"- Customer Details: {customer_context.get('inputs', {})}\n"
        )

    # Format previous turns if any
    history_text = ""
    if history:
        turns = []
        for h in history[-4:]:
            role_label = "User" if h.get("role") in ["user", "human"] else "Assistant"
            turns.append(f"{role_label}: {h.get('text', '')}")
        history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join(turns)

    full_prompt = (
        f"{SYSTEM_PROMPT}"
        f"{context_str}"
        f"{history_text}\n\n"
        f"User Question: {message}\n"
        f"Assistant Response:"
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini_raw, full_prompt, api_key)
            raw_text = future.result(timeout=timeout_seconds)

        if raw_text and raw_text.strip():
            return raw_text.strip(), "gemini"
        else:
            return generate_fallback_chat_reply(message, customer_context), "fallback"

    except (FutureTimeoutError, TimeoutError):
        logger.warning(f"Gemini chat call exceeded {timeout_seconds}s timeout. Using instant fallback reply.")
        return generate_fallback_chat_reply(message, customer_context), "fallback"
    except Exception as exc:
        logger.warning(f"Gemini chat invocation failed ({exc}). Using rule-based fallback.")
        return generate_fallback_chat_reply(message, customer_context), "fallback"


if __name__ == "__main__":
    reply, src = ask_churn_assistant("How to use the page?")
    print(f"Source: {src}")
    print(f"Reply:\n{reply}")
