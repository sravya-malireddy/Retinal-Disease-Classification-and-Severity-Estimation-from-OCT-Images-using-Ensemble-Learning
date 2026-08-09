import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.densenet import (
    preprocess_input as den_pre
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_pre
)

from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as res_pre
)

from Data_Preprocessing import (
    cfg,
    test_gen
)


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

MODEL_DIR = "/kaggle/working"

MODEL_PATHS = {
    "DenseNet": os.path.join(
        MODEL_DIR,
        "DenseNet.keras"
    ),
    "EffNet": os.path.join(
        MODEL_DIR,
        "EffNet.keras"
    ),
    "ResNet": os.path.join(
        MODEL_DIR,
        "ResNet.keras"
    )
}

ENSEMBLE_WEIGHTS = np.array(
    [0.3, 0.3, 0.4],
    dtype=np.float32
)


def load_trained_models():
    models = {}

    for name, path in MODEL_PATHS.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run Model_Training.py first."
            )

        print(f"Loading {name} model...")
        models[name] = load_model(path)

    return models


def tta_predict_batch(model, X, preprocess_func):
    predictions = []

    original = preprocess_func(
        (X * 255.0).astype(np.float32)
    )

    predictions.append(
        model.predict(
            original,
            verbose=0
        )
    )

    flipped = np.flip(
        X,
        axis=2
    )

    flipped = preprocess_func(
        (flipped * 255.0).astype(np.float32)
    )

    predictions.append(
        model.predict(
            flipped,
            verbose=0
        )
    )

    brighter = np.clip(
        X * 1.1,
        0.0,
        1.0
    )

    brighter = preprocess_func(
        (brighter * 255.0).astype(np.float32)
    )

    predictions.append(
        model.predict(
            brighter,
            verbose=0
        )
    )

    return np.mean(
        predictions,
        axis=0
    )


def get_predictions(
    model,
    generator,
    preprocess_func
):
    predictions = []

    for index in range(len(generator)):

        X, _ = generator[index]

        batch_predictions = tta_predict_batch(
            model,
            X,
            preprocess_func
        )

        predictions.append(
            batch_predictions
        )

    return np.vstack(predictions)


def get_true_labels(generator):
    labels = []

    for index in range(len(generator)):

        _, y = generator[index]

        labels.extend(
            np.argmax(y, axis=1)
        )

    return np.array(labels)


def evaluate_models(
    y_true,
    predictions
):
    dense_pred = np.argmax(
        predictions["DenseNet"],
        axis=1
    )

    eff_pred = np.argmax(
        predictions["EffNet"],
        axis=1
    )

    res_pred = np.argmax(
        predictions["ResNet"],
        axis=1
    )

    final_probabilities = (
        ENSEMBLE_WEIGHTS[0]
        * predictions["DenseNet"]
        +
        ENSEMBLE_WEIGHTS[1]
        * predictions["EffNet"]
        +
        ENSEMBLE_WEIGHTS[2]
        * predictions["ResNet"]
    )

    ensemble_pred = np.argmax(
        final_probabilities,
        axis=1
    )

    accuracies = {
        "DenseNet": accuracy_score(
            y_true,
            dense_pred
        ),

        "EffNet": accuracy_score(
            y_true,
            eff_pred
        ),

        "ResNet": accuracy_score(
            y_true,
            res_pred
        ),

        "Ensemble": accuracy_score(
            y_true,
            ensemble_pred
        )
    }

    print("\nModel Accuracy")
    print("-" * 40)

    for name, accuracy in accuracies.items():
        print(
            f"{name:10s}: "
            f"{accuracy:.4f}"
        )

    print("\nEnsemble Classification Report")
    print("-" * 40)

    print(
        classification_report(
            y_true,
            ensemble_pred,
            target_names=CLASS_NAMES
        )
    )

    confusion = confusion_matrix(
        y_true,
        ensemble_pred
    )

    print("Confusion Matrix")
    print("-" * 40)
    print(confusion)

    return (
        final_probabilities,
        ensemble_pred,
        confusion
    )


def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        try:
            if len(layer.output.shape) == 4:
                return layer.name

        except Exception:
            continue

    raise ValueError(
        "Could not find a suitable convolutional layer."
    )


def generate_gradcam(
    model,
    image
):
    last_conv_layer = find_last_conv_layer(
        model
    )

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(
                last_conv_layer
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_output, predictions = grad_model(
            image
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        loss = predictions[
            :,
            predicted_index
        ]

    gradients = tape.gradient(
        loss,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return (
        heatmap.numpy(),
        int(predicted_index.numpy()),
        predictions.numpy()[0]
    )


def generate_gradcam_sample(
    model,
    image_path
):
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image = cv2.resize(
        image,
        cfg.IMG_SIZE
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.stack(
        [image] * 3,
        axis=-1
    )

    input_image = np.expand_dims(
        image,
        axis=0
    )

    processed_image = res_pre(
        (input_image * 255.0).astype(
            np.float32
        )
    )

    heatmap, predicted_index, probabilities = (
        generate_gradcam(
            model,
            processed_image
        )
    )

    heatmap = cv2.resize(
        heatmap,
        cfg.IMG_SIZE
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    original_image = (
        image * 255
    ).astype(np.uint8)

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        np.max(probabilities)
    )

    return (
        overlay,
        predicted_class,
        confidence
    )


def main():

    print("Loading trained models...")
    models = load_trained_models()

    print("\nRunning Test-Time Augmentation...")

    predictions = {}

    predictions["DenseNet"] = get_predictions(
        models["DenseNet"],
        test_gen,
        den_pre
    )

    predictions["EffNet"] = get_predictions(
        models["EffNet"],
        test_gen,
        eff_pre
    )

    predictions["ResNet"] = get_predictions(
        models["ResNet"],
        test_gen,
        res_pre
    )

    y_true = get_true_labels(
        test_gen
    )

    final_probabilities, ensemble_predictions, confusion = (
        evaluate_models(
            y_true,
            predictions
        )
    )

    print("\nEnsemble evaluation completed.")

    return (
        models,
        final_probabilities,
        ensemble_predictions,
        confusion
    )


if __name__ == "__main__":
    main()
