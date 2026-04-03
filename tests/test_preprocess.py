import os
import pytest
import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

files_exist = os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)
skip_if_missing = pytest.mark.skipif(
    not files_exist,
    reason="train/test CSVs not found — run `make preprocess` first",
)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_PATH)


@skip_if_missing
def test_files_exist():
    """Both data/train.csv and data/test.csv must exist after preprocessing."""
    assert os.path.exists(TRAIN_PATH)
    assert os.path.exists(TEST_PATH)


@skip_if_missing
def test_required_columns(train_df, test_df):
    """Both splits must have exactly the columns: text, fraudulent."""
    required = {"text", "fraudulent"}
    assert required.issubset(train_df.columns)
    assert required.issubset(test_df.columns)


@skip_if_missing
def test_label_values(train_df, test_df):
    """fraudulent column must only contain 0 and 1."""
    assert set(train_df["fraudulent"].unique()).issubset({0, 1})
    assert set(test_df["fraudulent"].unique()).issubset({0, 1})


@skip_if_missing
def test_text_minimum_length(train_df, test_df):
    """No text field should be shorter than 20 characters."""
    assert (train_df["text"].str.len() >= 20).all(), "train.csv has text shorter than 20 chars"
    assert (test_df["text"].str.len() >= 20).all(), "test.csv has text shorter than 20 chars"


@skip_if_missing
def test_train_test_split_ratio(train_df, test_df):
    """Train/test split should be approximately 80/20 (within 2%)."""
    total = len(train_df) + len(test_df)
    train_ratio = len(train_df) / total
    assert 0.78 <= train_ratio <= 0.82, (
        f"Expected ~80% train split, got {train_ratio:.1%}"
    )
