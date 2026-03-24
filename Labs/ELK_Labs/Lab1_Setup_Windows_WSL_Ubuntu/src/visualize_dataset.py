import numpy as np
from tensorflow.keras.datasets import cifar10
from src.es_client import get_client, log_dataset_stats


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def compute_dataset_stats():
    """
    Loads CIFAR-10 and computes per-class statistics:
    - Sample count
    - Mean pixel intensity (R, G, B channels)
    - Std pixel intensity (R, G, B channels)
    - Overall mean brightness
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Combine train + test for dataset-level stats
    X_all = np.concatenate([X_train, X_test], axis=0).astype("float32") / 255.0
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()

    stats = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_all == cls_idx
        X_cls = X_all[mask]

        mean_r = float(X_cls[:, :, :, 0].mean())
        mean_g = float(X_cls[:, :, :, 1].mean())
        mean_b = float(X_cls[:, :, :, 2].mean())
        std_r  = float(X_cls[:, :, :, 0].std())
        std_g  = float(X_cls[:, :, :, 1].std())
        std_b  = float(X_cls[:, :, :, 2].std())

        stats.append({
            "class_id":          cls_idx,
            "class_name":        cls_name,
            "sample_count":      int(mask.sum()),
            "mean_r":            mean_r,
            "mean_g":            mean_g,
            "mean_b":            mean_b,
            "std_r":             std_r,
            "std_g":             std_g,
            "std_b":             std_b,
            "mean_brightness":   float((mean_r + mean_g + mean_b) / 3),
        })

        print(f"[dataset] {cls_name:12s} | n={mask.sum():6d} | "
              f"RGB mean=({mean_r:.3f}, {mean_g:.3f}, {mean_b:.3f})")

    return stats


def main():
    client = get_client()
    stats = compute_dataset_stats()
    log_dataset_stats(client, stats)
    print("[dataset] Dataset statistics indexed into Elasticsearch.")


if __name__ == "__main__":
    main()