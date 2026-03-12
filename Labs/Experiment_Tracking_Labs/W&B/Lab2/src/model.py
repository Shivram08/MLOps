from tensorflow import keras as k


def build_model(
    input_shape=(32, 32, 3),
    num_classes=10,
    layer_1_size=32,
    dropout=0.2,
    learn_rate=0.001,
):
    """
    3-block CNN for CIFAR-10 classification.

    Architecture:
        Block 1: Conv(layer_1_size)    -> BN -> MaxPool -> Dropout
        Block 2: Conv(layer_1_size*2)  -> BN -> MaxPool -> Dropout
        Block 3: Conv(layer_1_size*4)  -> BN -> MaxPool -> Dropout
        Head:    Flatten -> Dense(256) -> BN -> Dropout -> Dense(num_classes)

    Compared to the original lab (1 Conv block, SGD, no BN):
        - 3x deeper convolutional backbone
        - Batch normalization after every conv layer for stable training
        - Adam optimizer with cosine decay learning rate schedule
        - Larger dense head with its own BN + dropout

    Args:
        input_shape:  Shape of input images, default (32, 32, 3) for CIFAR-10.
        num_classes:  Number of output classes.
        layer_1_size: Number of filters in the first conv block.
                      Subsequent blocks double this (x2, x4).
        dropout:      Dropout rate applied after each pooling and dense layer.
        learn_rate:   Initial learning rate for Adam with cosine decay.

    Returns:
        Compiled Keras model.
    """
    inputs = k.Input(shape=input_shape)

    # Block 1
    x = k.layers.Conv2D(layer_1_size, (3, 3), padding="same", activation="relu")(inputs)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    # Block 2
    x = k.layers.Conv2D(layer_1_size * 2, (3, 3), padding="same", activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    # Block 3
    x = k.layers.Conv2D(layer_1_size * 4, (3, 3), padding="same", activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.MaxPooling2D((2, 2))(x)
    x = k.layers.Dropout(dropout)(x)

    # Classification head
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(256, activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Dropout(dropout)(x)
    outputs = k.layers.Dense(num_classes, activation="softmax")(x)

    model = k.Model(inputs, outputs)

    # Cosine decay schedule
    lr_schedule = k.optimizers.schedules.CosineDecay(
        initial_learning_rate=learn_rate,
        decay_steps=1000,
    )
    optimizer = k.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model