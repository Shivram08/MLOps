import numpy as np
import wandb
from tensorflow import keras as k


class LogLRCallback(k.callbacks.Callback):
    """Log the current learning rate to WandB each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr = opt.learning_rate
        # Handle both static float and LR schedule objects
        if hasattr(lr, "numpy"):
            lr_val = float(lr.numpy())
        elif hasattr(lr, "__call__"):
            lr_val = float(lr(opt.iterations).numpy())
        else:
            lr_val = float(lr)
        wandb.log({"lr": lr_val})


class LogSamplesCallback(k.callbacks.Callback):
    """
    Log a WandB table of sample predictions + images each epoch.
    Shows image, true label, predicted label, correctness, and confidence.
    """

    def __init__(self, x, y, labels, max_rows=32):
        super().__init__()
        self.x      = x[:max_rows]
        self.y      = y[:max_rows]
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        preds  = self.model.predict(self.x, verbose=0)
        y_true = np.argmax(self.y,     axis=1)
        y_pred = np.argmax(preds,      axis=1)

        table = wandb.Table(
            columns=["image", "y_true", "y_pred", "correct", "p(y_pred)"]
        )
        for i in range(len(self.x)):
            table.add_data(
                wandb.Image(self.x[i]),
                self.labels[y_true[i]],
                self.labels[y_pred[i]],
                bool(y_true[i] == y_pred[i]),
                float(np.max(preds[i])),
            )
        wandb.log({f"samples/epoch_{epoch + 1}": table})


class ConfusionMatrixCallback(k.callbacks.Callback):
    """Log a confusion matrix from the full validation set each epoch."""

    def __init__(self, x_val, y_val, labels):
        super().__init__()
        self.x_val  = x_val
        self.y_val  = y_val
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        preds  = self.model.predict(self.x_val, verbose=0)
        y_true = np.argmax(self.y_val, axis=1)
        y_pred = np.argmax(preds,      axis=1)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=self.labels,
            )
        })


class PerClassAccuracyCallback(k.callbacks.Callback):
    """
    Log per-class accuracy as a WandB bar chart each epoch.
    Not present in the original lab — shows which classes the model
    struggles with (e.g. cat vs dog on CIFAR-10).
    """

    def __init__(self, x_val, y_val, labels):
        super().__init__()
        self.x_val  = x_val
        self.y_val  = y_val
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        preds  = self.model.predict(self.x_val, verbose=0)
        y_true = np.argmax(self.y_val, axis=1)
        y_pred = np.argmax(preds,      axis=1)

        per_class = {}
        for cls_idx, cls_name in enumerate(self.labels):
            mask = y_true == cls_idx
            if mask.sum() > 0:
                per_class[f"per_class_acc/{cls_name}"] = float(
                    (y_pred[mask] == cls_idx).mean()
                )
        wandb.log(per_class)


class GradientHistogramCallback(k.callbacks.Callback):
    """
    Log weight gradient histograms for all trainable layers each epoch.
    Not present in the original lab — useful for diagnosing vanishing
    or exploding gradients in deeper networks.
    """

    def __init__(self, x_batch, y_batch):
        super().__init__()
        self.x_batch = x_batch
        self.y_batch = y_batch

    def on_epoch_end(self, epoch, logs=None):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            preds = self.model(self.x_batch, training=True)
            loss  = self.model.compiled_loss(self.y_batch, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)

        grad_logs = {}
        for var, grad in zip(self.model.trainable_variables, grads):
            if grad is not None:
                safe_name = var.name.replace(":", "_").replace("/", "_")
                grad_logs[f"gradients/{safe_name}"] = wandb.Histogram(
                    grad.numpy().flatten()
                )
        wandb.log(grad_logs)


class PRCurveCallback(k.callbacks.Callback):
    """
    Log per-class precision-recall curves each epoch.
    Not present in the original lab.
    """

    def __init__(self, x_val, y_val, labels):
        super().__init__()
        self.x_val  = x_val
        self.y_val  = y_val
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        preds  = self.model.predict(self.x_val, verbose=0)
        y_true = np.argmax(self.y_val, axis=1)

        wandb.log({
            "pr_curve": wandb.plot.pr_curve(
                y_true,
                preds,
                labels=self.labels,
            )
        })


def get_callbacks(X_test, y_test, labels):
    """
    Returns the full list of callbacks for a training run.
    Gradient histograms use the first batch of test data as a proxy.
    """
    x_batch = X_test[:64]
    y_batch = y_test[:64]

    return [
        LogLRCallback(),
        LogSamplesCallback(X_test, y_test, labels, max_rows=32),
        ConfusionMatrixCallback(X_test, y_test, labels),
        PerClassAccuracyCallback(X_test, y_test, labels),
        GradientHistogramCallback(x_batch, y_batch),
        PRCurveCallback(X_test, y_test, labels),
    ]