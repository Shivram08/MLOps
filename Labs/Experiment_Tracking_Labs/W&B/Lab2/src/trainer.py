import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from src.data import load_data, LABELS
from src.model import build_model
from src.callbacks import get_callbacks


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


DEFAULT_CONFIG = dict(
    dropout=0.2,
    layer_1_size=32,
    learn_rate=0.001,
    epochs=10,
    batch_size=64,
    sample=10000,       # set to None for full 50k training set
)


def _log_model_artifact(run, model):
    """Save model summary + weights and log as a WandB artifact."""
    os.makedirs("artifacts", exist_ok=True)

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    summary_path = "artifacts/model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    model_path = "artifacts/model.h5"
    model.save(model_path)

    art = wandb.Artifact("cifar10_cnn", type="model")
    art.add_file(summary_path)
    art.add_file(model_path)
    run.log_artifact(art)
    print(f"[trainer] Artifact logged: {model_path}")


def train(config=None, project="Lab2-cifar10-classification", run_name="cnn_run"):
    """
    Single training run. Can be called directly or from a sweep agent.

    When called from a sweep, wandb.init() is already initialised by the
    sweep agent and `config` is ignored — wandb.config is used instead.

    Args:
        config:   Dict of hyperparameters. Uses DEFAULT_CONFIG if None.
        project:  WandB project name.
        run_name: Display name for the run.
    """
    run = wandb.init(
        project=project,
        name=run_name,
        config=config or DEFAULT_CONFIG,
        settings=wandb.Settings(start_method="thread"),
    )
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, num_classes = load_data(sample=cfg.sample)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        input_shape=(32, 32, 3),
        num_classes=num_classes,
        layer_1_size=cfg.layer_1_size,
        dropout=cfg.dropout,
        learn_rate=cfg.learn_rate,
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        WandbMetricsLogger(log_freq=10),
        WandbModelCheckpoint(
            "checkpoints/model-{epoch:02d}.h5",
            save_weights_only=False,
        ),
        *get_callbacks(X_test, y_test, LABELS),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Final eval ────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    wandb.log({"final/loss": loss, "final/accuracy": acc})
    print(f"[trainer] Final — loss: {loss:.4f}, accuracy: {acc:.4f}")

    _log_model_artifact(run, model)
    run.finish()