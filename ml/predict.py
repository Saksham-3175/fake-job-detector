import joblib

# Load trained pipeline
model = joblib.load("models/fake_job_model.joblib")


def predict_listing(
    title: str,
    description: str,
    company_profile: str = "",
    requirements: str = "",
) -> dict:
    """Predict whether a job listing is real or fake."""
    # Combine inputs in the same format as training
    text = f"{title} {company_profile} {description} {requirements}"

    # Get prediction and probabilities
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    real_prob = float(probabilities[0])
    fake_prob = float(probabilities[1])
    is_fake = bool(prediction == 1)
    confidence = float(max(real_prob, fake_prob))

    # Determine trust level
    if confidence > 0.85:
        trust_level = "High"
    elif confidence > 0.65:
        trust_level = "Medium"
    else:
        trust_level = "Low"

    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "verdict": "FAKE" if is_fake else "REAL",
        "trust_level": trust_level,
    }


def batch_predict(listings: list[dict]) -> list[dict]:
    """Predict on a batch of listings and enrich each with prediction results."""
    results = []
    for listing in listings:
        prediction = predict_listing(
            title=listing.get("title", ""),
            description=listing.get("description", ""),
            company_profile=listing.get("company_profile", ""),
            requirements=listing.get("requirements", ""),
        )
        enriched = {**listing, **prediction}
        results.append(enriched)
    return results


if __name__ == "__main__":
    # Test examples
    examples = [
        {
            "label": "Example 1 (likely real)",
            "title": "Software Engineer Intern",
            "description": (
                "We are looking for a software engineering intern to join our team "
                "at our Bangalore office. You will work with Python, React and FastAPI. "
                "Requirements: B.Tech CS, knowledge of data structures."
            ),
            "company_profile": "Tech startup based in Bangalore, founded 2019.",
        },
        {
            "label": "Example 2 (likely fake)",
            "title": "Work From Home Data Entry Jobs - Earn 50000 Monthly",
            "description": (
                "No experience needed. Earn money from home. Send your details to claim "
                "your job. Guaranteed income. Start immediately. No investment required."
            ),
            "company_profile": "",
        },
        {
            "label": "Example 3 (ambiguous)",
            "title": "Marketing Executive",
            "description": (
                "Immediate opening for marketing executive. Good communication skills "
                "required. Contact HR on WhatsApp. Salary negotiable."
            ),
            "company_profile": "Growing company",
        },
    ]

    for ex in examples:
        label = ex.pop("label")
        result = predict_listing(
            title=ex["title"],
            description=ex["description"],
            company_profile=ex.get("company_profile", ""),
        )
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"  Title:       {ex['title']}")
        print(f"  Verdict:     {result['verdict']}")
        print(f"  Confidence:  {result['confidence']:.2%}")
        print(f"  Trust Level: {result['trust_level']}")
        print(f"  Real Prob:   {result['real_probability']:.4f}")
        print(f"  Fake Prob:   {result['fake_probability']:.4f}")
