"""Automated unit and integration test suite for ChurnSense API."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient with active FastAPI lifespan."""
    with TestClient(app) as c:
        yield c

SAMPLE_VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 53.85,
    "TotalCharges": 646.20,
}


def test_health_endpoint(client):
    """Verify GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ChurnSense"


def test_predict_valid_customer(client):
    """Verify POST /predict returns probability, verdict and threshold."""
    response = client.post("/predict", json=SAMPLE_VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in ["Will churn", "Will stay"]
    assert "threshold" in data
    assert data["threshold"] > 0


def test_predict_invalid_input_422(client):
    """Verify bad inputs trigger HTTP 422 unprocessable entity with field details."""
    bad_payload = SAMPLE_VALID_CUSTOMER.copy()
    bad_payload["tenure"] = -5  # invalid tenure < 0
    bad_payload["Contract"] = "Infinite year"  # invalid Literal

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "details" in data
    assert len(data["details"]) >= 1


def test_explain_valid_customer_mocked_gemini(client):
    """Verify POST /explain returns prediction fields, top SHAP reasons, and LLM explanation."""
    mock_explanation = "This customer is likely to stay due to their one-year contract and active technical support."

    with patch("api.explain", return_value=(mock_explanation, "gemini")):
        response = client.post("/explain", json=SAMPLE_VALID_CUSTOMER)
        assert response.status_code == 200
        data = response.json()

        assert "churn_probability" in data
        assert "prediction" in data
        assert "threshold" in data
        assert "top_reasons" in data
        assert len(data["top_reasons"]) == 3
        for reason in data["top_reasons"]:
            assert "feature" in reason
            assert "value" in reason
            assert "impact" in reason
            assert "direction" in reason
            assert reason["direction"] in ["increases churn risk", "decreases churn risk"]

        assert data["explanation"] == mock_explanation
        assert data["explanation_source"] == "gemini"


def test_explain_fallback(client):
    """Verify POST /explain gracefully falls back if LLM generates fallback text."""
    fallback_text = "This customer is predicted to will stay with a 15.0% churn probability."

    with patch("api.explain", return_value=(fallback_text, "fallback")):
        response = client.post("/explain", json=SAMPLE_VALID_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert data["explanation_source"] == "fallback"
        assert data["explanation"] == fallback_text


def test_root_serves_html(client):
    """Verify GET / returns 200 and serves HTML containing 'ChurnSense'."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ChurnSense" in response.text


def test_chat_endpoint_valid_question(client):
    """Verify POST /chat responds to queries with text and source indicator."""
    mock_reply = "Month-to-month contracts lack long-term retention lock-in."
    with patch("api.ask_churn_assistant", return_value=(mock_reply, "gemini")):
        response = client.post("/chat", json={"message": "Why is contract type important?"})
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == mock_reply
        assert data["source"] == "gemini"


def test_chat_endpoint_with_context(client):
    """Verify POST /chat accepts customer context and history."""
    mock_reply = "Based on this customer's 88% probability, offer a 1-year contract discount."
    with patch("api.ask_churn_assistant", return_value=(mock_reply, "gemini")):
        payload = {
            "message": "What offer should we make to this customer?",
            "history": [
                {"role": "user", "text": "Is this customer high risk?"},
                {"role": "assistant", "text": "Yes, 88% probability."}
            ],
            "customer_context": {
                "churn_probability": 0.88,
                "prediction": "Will churn",
                "top_reasons": [{"feature": "Contract: Month-to-month", "value": "Month-to-month", "direction": "increases churn risk"}]
            }
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == mock_reply


def test_chat_endpoint_validation_error(client):
    """Verify empty chat message triggers 422 validation error."""
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_models_endpoint(client):
    """Verify GET /models returns list of all candidate models."""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 3
    ids = [m["id"] for m in models]
    assert "logistic_regression" in ids
    assert "random_forest" in ids
    assert "xgboost" in ids


def test_demo_cases_endpoint(client):
    """Verify GET /demo-cases returns the 3 canonical benchmark personas."""
    response = client.get("/demo-cases")
    assert response.status_code == 200
    cases = response.json()
    assert len(cases) == 3
    ids = [c["id"] for c in cases]
    assert "high-risk" in ids
    assert "low-risk" in ids
    assert "borderline" in ids


def test_predict_with_model_param(client):
    """Verify POST /predict works with model parameter and returns consensus."""
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        response = client.post(f"/predict?model={model_name}", json=SAMPLE_VALID_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == model_name
        assert "consensus" in data
        assert isinstance(data["consensus"], dict)
