from elasticsearch import Elasticsearch
from datetime import datetime
import time


ES_HOST = "http://localhost:9200"
INDICES = {
    "training": "cifar10-training-metrics",
    "predictions": "cifar10-predictions",
    "dataset":  "cifar10-dataset-stats",
}


def get_client(retries=10, delay=5):
    """
    Returns an Elasticsearch client, retrying until ES is ready.
    Useful when running after docker compose up.
    """
    for attempt in range(retries):
        try:
            client = Elasticsearch(ES_HOST)
            if client.ping():
                print(f"[es_client] Connected to Elasticsearch at {ES_HOST}")
                return client
        except Exception:
            pass
        print(f"[es_client] Waiting for Elasticsearch... ({attempt + 1}/{retries})")
        time.sleep(delay)
    raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")


def log_training_metrics(client, epoch, loss, accuracy, val_loss, val_accuracy):
    """
    Index one document per epoch into cifar10-training-metrics.
    """
    doc = {
        "timestamp": datetime.utcnow().isoformat(),
        "epoch": epoch,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
    }
    client.index(index=INDICES["training"], body=doc)


def log_predictions(client, y_true, y_pred, y_proba, class_names):
    """
    Index one document per test sample into cifar10-predictions.
    Stores true label, predicted label, confidence, and correctness.
    """
    docs = []
    for i in range(len(y_true)):
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "sample_id": i,
            "true_label": class_names[y_true[i]],
            "predicted_label": class_names[y_pred[i]],
            "confidence": float(y_proba[i]),
            "correct": bool(y_true[i] == y_pred[i]),
        }
        docs.append(doc)

    # Bulk index for speed
    from elasticsearch.helpers import bulk
    actions = [
        {"_index": INDICES["predictions"], "_source": doc}
        for doc in docs
    ]
    bulk(client, actions)
    print(f"[es_client] Indexed {len(docs)} predictions")


def log_dataset_stats(client, stats):
    """
    Index dataset statistics into cifar10-dataset-stats.
    stats is a list of dicts, one per class.
    """
    from elasticsearch.helpers import bulk
    actions = [
        {"_index": INDICES["dataset"], "_source": {**s, "timestamp": datetime.utcnow().isoformat()}}
        for s in stats
    ]
    bulk(client, actions)
    print(f"[es_client] Indexed {len(stats)} dataset stat documents")