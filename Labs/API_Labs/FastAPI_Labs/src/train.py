import json
import os
from datetime import datetime

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_dataset

MODEL_DIR = os.path.join("..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "penguins_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
MODEL_CARD_PATH = os.path.join(MODEL_DIR, "model_card.json")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, feature_names, class_names = load_dataset()

    # Identify numeric vs categorical columns based on dtype
    numeric_features = [c for c in X.columns if X[c].dtype.kind in ("i", "u", "f")]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    model_card = {
        "model_name": "penguins-logreg-pipeline",
        "task": "multiclass_classification",
        "classes": class_names,
        "features": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("Model card saved to:", MODEL_CARD_PATH)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
