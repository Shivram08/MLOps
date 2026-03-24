import os
import logging
import numpy as np
from datetime import datetime
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow import keras as k
from sklearn.metrics import f1_score, confusion_matrix

from src.es_client import get_client, log_training_metrics, log_predictions

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logstash", exist_ok=True)
logging.basicConfig(
    filename="logstash/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

SAMPLE = 10000   # set to None for full 50k dataset


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(sample=SAMPLE):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    if sample:
        X_train, y_train = X_train[:sample], y_train[:sample]
        X_test,  y_test  = X_test[:sample],  y_test[:sample]
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)
    return X_train, X_test, y_train.flatten(), y_test.flatten(), y_train_cat, y_test_cat


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(dropout=0.2, layer_1_size=32, learn_rate=0.001):
    inputs = k.Input(shape=(32, 32, 3))

    x = k.layers.Conv2D(layer_1_size, (3, 3), padding="same", activation="relu")(inputs)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    x = k.layers.Conv2D(layer_1_size * 2, (3, 3), padding="same", activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    x = k.layers.Conv2D(layer_1_size * 4, (3, 3), padding="same", activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    x = k.layers.Flatten()(x)
    x = k.layers.Dense(256, activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Dropout(dropout)(x)
    outputs = k.layers.Dense(10, activation="softmax")(x)

    model = k.Model(inputs, outputs)
    lr_schedule = k.optimizers.schedules.CosineDecay(
        initial_learning_rate=learn_rate, decay_steps=1000
    )
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── ELK Callback ─────────────────────────────────────────────────────────────
class ELKLoggerCallback(k.callbacks.Callback):
    """
    Logs per-epoch metrics to:
      1. training.log file  (picked up by Logstash)
      2. Elasticsearch directly via es_client
    """
    def __init__(self, es_client):
        super().__init__()
        self.es = es_client

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss     = logs.get("loss", 0)
        acc      = logs.get("accuracy", 0)
        val_loss = logs.get("val_loss", 0)
        val_acc  = logs.get("val_accuracy", 0)

        # Log to file for Logstash
        logging.info(f"Epoch: {epoch + 1}")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info(f"Val Accuracy: {val_acc:.4f}")

        # Log directly to Elasticsearch
        log_training_metrics(
            self.es, epoch + 1, loss, acc, val_loss, val_acc
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.info("=" * 50)
    logging.info(f"Training started at {datetime.utcnow().isoformat()}")

    # Connect to Elasticsearch
    es = get_client()

    # Load data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_data()
    logging.info(f"Dataset: CIFAR-10 | Train: {len(X_train)} | Test: {len(X_test)}")

    # Build model
    model = build_model(dropout=0.2, layer_1_size=32, learn_rate=0.001)
    model.summary()

    # Train
    logging.info("Starting model training...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=10,
        batch_size=64,
        callbacks=[ELKLoggerCallback(es)],
        verbose=1,
    )
    logging.info("Model training completed.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)
    y_conf  = np.max(y_proba, axis=1)

    # Metrics
    score = float(np.mean(y_pred == y_test))
    f1    = f1_score(y_test, y_pred, average="weighted")
    cm    = confusion_matrix(y_test, y_pred)

    tp = np.diag(cm)
    tn = np.sum(cm) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - tp)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    fp_rate = np.divide(fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) != 0)
    fn_rate = np.divide(fn, fn + tp, out=np.zeros_like(fn, dtype=float), where=(fn + tp) != 0)

    # Log evaluation to file
    logging.info(f"Model Accuracy: {score:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"True Positive: {tp.tolist()}")
    logging.info(f"True Negative: {tn.tolist()}")
    logging.info(f"False Positive: {fp.tolist()}")
    logging.info(f"False Negative: {fn.tolist()}")
    logging.info(f"False Positive Rate: {fp_rate.tolist()}")
    logging.info(f"False Negative Rate: {fn_rate.tolist()}")

    # Per-class accuracy
    for i, cls in enumerate(CLASS_NAMES):
        mask     = y_test == i
        cls_acc  = float(np.mean(y_pred[mask] == y_test[mask]))
        logging.info(f"Per-class accuracy [{cls}]: {cls_acc:.4f}")

    # Index predictions to Elasticsearch
    log_predictions(es, y_test, y_pred, y_conf, CLASS_NAMES)

    logging.info(f"Training finished at {datetime.utcnow().isoformat()}")
    logging.info("=" * 50)
    print(f"\nFinal accuracy: {score:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    main()