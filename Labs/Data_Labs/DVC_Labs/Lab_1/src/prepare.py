import os
import re
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def clean_columns(cols):
    cleaned = []
    for c in cols:
        c2 = c.strip()
        c2 = re.sub(r"\s+", "_", c2)
        cleaned.append(c2)
    return cleaned


def main():
    params = yaml.safe_load(open("params.yaml", "r"))
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    df = pd.read_csv("data/student_performance.csv")
    df.columns = clean_columns(df.columns)

    target = "exam_score"
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)[:10]}..."
        )

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # One-hot encode categoricals
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Prepared data saved to data/processed/")


if __name__ == "__main__":
    main()
