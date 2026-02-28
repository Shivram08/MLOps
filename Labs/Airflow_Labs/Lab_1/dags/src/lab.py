import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ──────────────────────────────────────────────
# Task 1: Load Data
# ──────────────────────────────────────────────
def load_data():
    """
    Loads the Heart Disease dataset from CSV, serializes it,
    and returns it for downstream tasks via XCom.
    """
    df = pd.read_csv("/opt/airflow/dags/data/heart.csv")
    print(f"[load_data] Loaded dataset with shape: {df.shape}")
    return pickle.dumps(df)


# ──────────────────────────────────────────────
# Task 2: Preprocess Data
# ──────────────────────────────────────────────
def preprocess_data(data):
    """
    Deserializes the raw dataframe, encodes categorical columns,
    scales numeric features, and returns a train/test split
    as a serialized dictionary.
    """
    df = pickle.loads(data)

    # Drop non-predictive identifier columns
    df = df.drop(columns=["id", "dataset"], errors="ignore")

    # Binarize target: 0 = no disease, 1 = disease (num values 1-4)
    df["num"] = (df["num"] > 0).astype(int)

    # Drop rows with any missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=["num"])
    y = df["num"]

    # One-hot encode categorical columns
    cat_cols = ["sex", "cp", "restecg", "exang", "slope", "thal"]
    X = pd.get_dummies(X, columns=cat_cols)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (80/20, stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[preprocess_data] Train size: {len(X_train)}, Test size: {len(X_test)}")

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_test": y_test.values,
    }
    return pickle.dumps(payload)


# ──────────────────────────────────────────────
# Task 3: Train Model
# ──────────────────────────────────────────────
def train_model(data, filename):
    """
    Deserializes the preprocessed data, trains a Random Forest
    classifier, saves the model to disk, and returns the
    serialized test split for evaluation.
    """
    payload = pickle.loads(data)
    X_train = payload["X_train"]
    y_train = payload["y_train"]

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",   # handles mild class imbalance
    )
    clf.fit(X_train, y_train)

    # Save model to disk
    model_path = f"/opt/airflow/working_data/{filename}"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"[train_model] Model saved to {model_path}")

    # Pass test split downstream for evaluation
    test_payload = {
        "X_test": payload["X_test"],
        "y_test": payload["y_test"],
        "model_path": model_path,
    }
    return pickle.dumps(test_payload)


# ──────────────────────────────────────────────
# Task 4: Evaluate Model
# ──────────────────────────────────────────────
def evaluate_model(data):
    """
    Loads the saved model, runs inference on the held-out test set,
    computes classification metrics, and writes a report to disk.
    """
    payload = pickle.loads(data)
    X_test = payload["X_test"]
    y_test = payload["y_test"]
    model_path = payload["model_path"]

    # Load model from disk
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Predict
    y_pred = clf.predict(X_test)

    # Compute metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    report_lines = [
        "=" * 40,
        "Heart Disease Classification Report",
        "=" * 40,
        f"Accuracy : {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall   : {recall:.4f}",
        f"F1 Score : {f1:.4f}",
        "=" * 40,
    ]

    report = "\n".join(report_lines) + "\n"
    for line in report_lines:
        print(line)

    # Save report to working_data so it persists after the container stops
    report_path = "/opt/airflow/working_data/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"[evaluate_model] Report saved to {report_path}")
    return report