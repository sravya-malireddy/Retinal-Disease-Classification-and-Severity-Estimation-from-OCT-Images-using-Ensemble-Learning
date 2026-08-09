import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB4,
    ResNet50V2
)

from tensorflow.keras.applications.densenet import (
    preprocess_input as den_pre
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_pre
)

from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as res_pre
)

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    Multiply
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.regularizers import l2

from sklearn.utils.class_weight import compute_class_weight

from Data_Preprocessing import (
    cfg,
    train_gen,
    val_gen
)


INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 4


class PreprocessWrapper(tf.keras.utils.Sequence):

    def __init__(self, generator, preprocess_func):
        self.generator = generator
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        X, y = self.generator[index]

        X = self.preprocess_func(
            (X * 255.0).astype(np.float32)
        )

        return X, y

    def on_epoch_end(self):
        self.generator.on_epoch_end()


def se_block(x):
    channels = int(x.shape[-1])

    se = Dense(
        channels // 8,
        activation="relu"
    )(x)

    se = Dense(
        channels,
        activation="sigmoid"
    )(se)

    return Multiply()([x, se])


def build_model(base_fn):

    base = base_fn(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    base.trainable = False

    x = base.output

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(
        1024,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(0.5)(x)

    x = Dense(
        512,
        activation="relu"
    )(x)

    x = Dropout(0.3)(x)

    x = se_block(x)

    output = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base.input,
        outputs=output
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        metrics=["accuracy"]
    )

    return model, base


def calculate_class_weights(generator):

    labels = []

    for index in range(len(generator)):
        _, y = generator[index]

        labels.extend(
            np.argmax(y, axis=1)
        )

    labels = np.array(labels)

    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    return dict(
        zip(classes, weights)
    )


class_weights = calculate_class_weights(train_gen)


def train_model(
    name,
    base_fn,
    preprocess_fn,
    unfreeze_n
):

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    model, base = build_model(base_fn)

    train_wrapper = PreprocessWrapper(
        train_gen,
        preprocess_fn
    )

    val_wrapper = PreprocessWrapper(
        val_gen,
        preprocess_fn
    )

    checkpoint_path = f"/kaggle/working/{name}.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2
        ),

        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print(f"\n{name}: Initial training")

    model.fit(
        train_wrapper,
        validation_data=val_wrapper,
        epochs=8,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n{name}: Fine-tuning")

    base.trainable = True

    for layer in base.layers[:-unfreeze_n]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        metrics=["accuracy"]
    )

    model.fit(
        train_wrapper,
        validation_data=val_wrapper,
        epochs=25,
        initial_epoch=8,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n{name} training completed.")
    print(f"Best model saved to: {checkpoint_path}")

    return model


models_config = {
    "DenseNet": (
        DenseNet121,
        den_pre,
        120
    ),

    "EffNet": (
        EfficientNetB4,
        eff_pre,
        150
    ),

    "ResNet": (
        ResNet50V2,
        res_pre,
        100
    )
}


def main():

    print("Starting model training...")

    print("\nClasses:")
    for index, class_name in enumerate(cfg.CLASSES):
        print(f"{index}: {class_name}")

    print("\nClass weights:")
    for class_id, weight in class_weights.items():
        print(
            f"{cfg.CLASSES[int(class_id)]}: "
            f"{weight:.4f}"
        )

    trained_models = {}

    for name, (
        model_function,
        preprocess_function,
        unfreeze_layers
    ) in models_config.items():

        trained_models[name] = train_model(
            name,
            model_function,
            preprocess_function,
            unfreeze_layers
        )

    print("\nAll models have been trained.")

    return trained_models


if __name__ == "__main__":
    main()
