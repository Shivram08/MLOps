import json
import os
from typing import Dict, List, Any

import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "penguins_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
MODEL_CARD_PATH = os.path.join(MODEL_DIR, "model_card.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with: python train.py"
        )
    return joblib.load(MODEL_PATH)


def predict_one(model, features: Dict[str, Any]) -> str:
    """
    features example:
      {
        "flipper_length_mm": 190,
        "body_mass_g": 4200,
        "island": "Biscoe",
        "sex": "MALE"
      }
    """
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    return str(pred)


def predict_proba_one(model, features: Dict[str, Any]) -> Dict[str, float]:
    X = pd.DataFrame([features])

    # Some models/pipelines may not implement predict_proba
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba.")

    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)
    return {str(c): float(p) for c, p in zip(classes, probs)}


def explain_one(model, features: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Lightweight explainability:
    - For linear models inside a Pipeline: we can approximate contributions as coef * feature_value
      (only for numeric features; categorical one-hot is harder without full mapping).
    - If we can't access coefficients, we return a helpful message.
    """
    try:
        preprocess = model.named_steps["preprocess"]
        clf = model.named_steps["classifier"]
    except Exception:
        return {"note": "Explainability not available for this model structure."}

    if not hasattr(clf, "coef_"):
        return {"note": "Model does not expose coefficients for explanation."}

    # Build a single-row DataFrame
    X_df = pd.DataFrame([features])

    # Transform input
    X_trans = preprocess.transform(X_df)

    # coef_ shape: (n_classes, n_features) for multiclass, or (1, n_features) for binary
    coefs = clf.coef_
    classes = list(model.classes_)

    # Pick the predicted class row for multiclass, else row 0
    pred_label = model.predict(X_df)[0]
    if len(coefs.shape) == 2 and coefs.shape[0] > 1:
        class_idx = classes.index(pred_label)
        coef_vec = coefs[class_idx]
    else:
        coef_vec = coefs[0]

    # Compute contributions in transformed feature space
    contrib = (X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans)[0] * coef_vec

    # Get transformed feature names if possible
    try:
        feat_names = preprocess.get_feature_names_out()
        feat_names = [str(f) for f in feat_names]
    except Exception:
        feat_names = [f"feature_{i}" for i in range(len(contrib))]

    pairs = sorted(
        [{"feature": feat_names[i], "contribution": float(contrib[i])} for i in range(len(contrib))],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )[: max(1, int(top_k))]

    return {
        "predicted_class": str(pred_label),
        "top_contributions": pairs,
        "note": "Contributions are coef * transformed_feature_value (approximate).",
    }


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metrics() -> Dict[str, Any]:
    return read_json(METRICS_PATH)


def get_model_card() -> Dict[str, Any]:
    return read_json(MODEL_CARD_PATH)
