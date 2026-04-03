import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load dataset
    df = pd.read_csv("data/fake_job_postings.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['fraudulent'].value_counts()}")

    # Fill NaN with empty string in text columns
    text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
    for col in text_columns:
        df[col] = df[col].fillna("")

    # Combine text columns
    df["text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    )

    # Keep only text and fraudulent columns
    df = df[["text", "fraudulent"]]

    # Drop rows where text is empty or less than 20 characters
    df = df[df["text"].str.strip().str.len() >= 20]

    # Print final class distribution
    real_count = (df["fraudulent"] == 0).sum()
    fake_count = (df["fraudulent"] == 1).sum()
    fake_pct = fake_count / len(df) * 100

    print("\nFinal class distribution:")
    print(f"  Real listings: {real_count}")
    print(f"  Fake listings: {fake_count}")
    print(f"  Fake percentage: {fake_pct:.2f}%")

    # Split into train and test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["fraudulent"],
        random_state=42,
    )

    # Save to CSV
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"\nPreprocessing complete. Train: {len(train_df)} rows, Test: {len(test_df)} rows")


if __name__ == "__main__":
    main()
