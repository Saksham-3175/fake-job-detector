import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import mlflow
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df["text"]
    y_train = train_df["fraudulent"]
    X_test = test_df["text"]
    y_test = test_df["fraudulent"]

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # Set up MLflow
    mlflow.set_experiment("fake-job-detector")

    with mlflow.start_run(run_name="tfidf-logistic-regression"):
        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_features", 50000)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("class_weight", "balanced")

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Real", "Fake"], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
        print(f"Confusion Matrix:\n{cm}")

        # Save metrics to file for DVC tracking
        metrics = {
            "accuracy": accuracy,
            "precision_fake": report["Fake"]["precision"],
            "recall_fake": report["Fake"]["recall"],
            "f1_fake": report["Fake"]["f1-score"],
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_fake", report["Fake"]["precision"])
        mlflow.log_metric("recall_fake", report["Fake"]["recall"])
        mlflow.log_metric("f1_fake", report["Fake"]["f1-score"])
        mlflow.log_metric("precision_real", report["Real"]["precision"])
        mlflow.log_metric("recall_real", report["Real"]["recall"])
        mlflow.log_metric("f1_real", report["Real"]["f1-score"])

        # Generate confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("models/confusion_matrix.png", dpi=150)
        plt.close()

        # Log confusion matrix image as artifact
        mlflow.log_artifact("models/confusion_matrix.png")

        # Save model
        joblib.dump(pipeline, "models/fake_job_model.joblib")

        # Log model artifact
        mlflow.log_artifact("models/fake_job_model.joblib")

    # Print summary
    recall_fake = report["Fake"]["recall"] * 100
    print(f"\nTraining complete.")
    print(f"Model saved to models/fake_job_model.joblib")
    print(f"Recall on fake listings: {recall_fake:.1f}% (higher = better at catching fakes)")
    print(f"Open MLflow UI with: mlflow ui")


if __name__ == "__main__":
    main()
