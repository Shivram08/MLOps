import os
import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor


def main():
    params = yaml.safe_load(open("params.yaml", "r"))

    random_state = params["train"]["random_state"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    min_samples_split = params["model"]["min_samples_split"]

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
