import os
import pandas as pd


def load_dataset():
    """
    Loads penguins.csv and returns:
      X: pandas.DataFrame (features)
      y: pandas.Series (target: species)
      feature_names: list[str]
      class_names: list[str]
    """
    base_dir = os.path.dirname(__file__)  # .../src
    csv_path = os.path.join(base_dir, "..", "data", "penguins.csv")

    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    if "species" not in df.columns:
        raise ValueError("penguins.csv must contain a 'species' column.")

    y = df["species"]

    # We include numeric features, and add island/sex if available
    expected = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island", "sex"]
    feature_cols = [c for c in expected if c in df.columns]

    if not feature_cols:
        raise ValueError(f"No expected feature columns found. Expected one of: {expected}")

    X = df[feature_cols]
    feature_names = feature_cols
    class_names = sorted(y.unique().tolist())

    return X, y, feature_names, class_names
