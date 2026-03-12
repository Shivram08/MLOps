import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_data(sample: int = None):
    """
    Loads CIFAR-10, normalizes pixel values to [0, 1],
    and one-hot encodes labels.

    Args:
        sample: If set, limits both train and test to this many samples.
                Useful for fast sweeps. If None, uses the full dataset.

    Returns:
        X_train, X_test, y_train, y_test, num_classes
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Optionally subsample for faster sweep runs
    if sample is not None:
        X_train = X_train[:sample]
        y_train = y_train[:sample]
        X_test  = X_test[:sample]
        y_test  = y_test[:sample]

    # One-hot encode
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test,  num_classes)

    print(f"[data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, num_classes