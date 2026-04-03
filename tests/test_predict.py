import pytest
from ml.predict import predict_listing

EXPECTED_KEYS = {
    "is_fake",
    "confidence",
    "real_probability",
    "fake_probability",
    "verdict",
    "trust_level",
}

FAKE_TEXT = {
    "title": "Work from home earn 50000 no experience",
    "description": "Send details immediately guaranteed income no investment required",
    "company_profile": "",
    "requirements": "",
}

REAL_TEXT = {
    "title": "Software Engineer at Google, Mountain View",
    "description": (
        "Requirements: BS in CS, 3+ years Python, system design experience. "
        "Benefits: health insurance, 401k."
    ),
    "company_profile": "Google LLC, Mountain View CA.",
    "requirements": "BS Computer Science, 3+ years Python, system design.",
}


@pytest.fixture(scope="module")
def fake_result():
    return predict_listing(**FAKE_TEXT)


@pytest.fixture(scope="module")
def real_result():
    return predict_listing(**REAL_TEXT)


def test_schema(fake_result):
    """predict_listing output contains all required keys."""
    assert EXPECTED_KEYS.issubset(fake_result.keys())


def test_verdict_values(fake_result, real_result):
    """verdict is always either REAL or FAKE."""
    assert fake_result["verdict"] in ("REAL", "FAKE")
    assert real_result["verdict"] in ("REAL", "FAKE")


def test_confidence_range(fake_result, real_result):
    """confidence is a probability between 0 and 1."""
    assert 0.0 <= fake_result["confidence"] <= 1.0
    assert 0.0 <= real_result["confidence"] <= 1.0


def test_probabilities_sum_to_one(fake_result, real_result):
    """real_probability + fake_probability ≈ 1.0."""
    assert abs(fake_result["real_probability"] + fake_result["fake_probability"] - 1.0) < 1e-6
    assert abs(real_result["real_probability"] + real_result["fake_probability"] - 1.0) < 1e-6


def test_obviously_fake_prediction(fake_result):
    """Obvious scam listing should be predicted as FAKE."""
    assert fake_result["verdict"] == "FAKE"


def test_obviously_real_prediction(real_result):
    """Legitimate Google job listing should be predicted as REAL."""
    assert real_result["verdict"] == "REAL"


def test_empty_input_does_not_crash():
    """Empty string inputs should return a valid result without raising."""
    result = predict_listing(title="", description="", company_profile="", requirements="")
    assert EXPECTED_KEYS.issubset(result.keys())
    assert result["verdict"] in ("REAL", "FAKE")


def test_trust_level_values(fake_result, real_result):
    """trust_level must be one of the three defined levels."""
    valid = {"High", "Medium", "Low"}
    assert fake_result["trust_level"] in valid
    assert real_result["trust_level"] in valid
