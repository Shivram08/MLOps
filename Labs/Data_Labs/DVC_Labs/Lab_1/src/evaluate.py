import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score


def main():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    metrics = {"rmse": float(rmse), "r2": float(r2)}

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    main()
