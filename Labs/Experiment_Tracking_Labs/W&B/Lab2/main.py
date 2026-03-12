from src.trainer import train, DEFAULT_CONFIG

if __name__ == "__main__":
    train(
        config=DEFAULT_CONFIG,
        project="Lab2-cifar10-classification",
        run_name="cnn_baseline",
    )