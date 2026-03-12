import wandb
import yaml
from src.trainer import train


def run_sweep(count=10):
    """
    Initialises a WandB sweep from sweep_config.yaml and runs `count` trials.

    Each trial calls train() with hyperparameters sampled by the sweep agent.
    Results are visible in the WandB sweep dashboard.

    Args:
        count: Number of sweep trials to run.
    """
    with open("sweep_config.yaml", "r") as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep_cfg,
        project="Lab2-cifar10-classification",
    )
    print(f"[sweep] Sweep ID: {sweep_id}")

    # train() detects it's inside a sweep via wandb.config
    wandb.agent(sweep_id, function=train, count=count)


if __name__ == "__main__":
    run_sweep(count=10)