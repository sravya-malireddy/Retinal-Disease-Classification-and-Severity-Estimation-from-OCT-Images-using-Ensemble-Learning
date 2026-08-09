import os
import glob
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


class Config:
    BASE_PATH = "/kaggle/input/datasets/obulisainaren/retinal-oct-c8/RetinalOCT_Dataset/RetinalOCT_Dataset"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32
    CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
    RANDOM_STATE = 42


cfg = Config()

random.seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)
tf.random.set_seed(cfg.RANDOM_STATE)


def load_data(base_path, split):
    paths = []
    labels = []

    split_path = os.path.join(base_path, split)

    if not os.path.isdir(split_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {split_path}"
        )

    for class_name in cfg.CLASSES:
        class_folder = os.path.join(split_path, class_name)

        if not os.path.isdir(class_folder):
            print(f"Warning: class folder not found: {class_folder}")
            continue

        image_paths = []

        for extension in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(
                glob.glob(os.path.join(class_folder, extension))
            )

        paths.extend(image_paths)
        labels.extend([class_name] * len(image_paths))

    return paths, labels


def create_dataframes():
    train_paths, train_labels = load_data(cfg.BASE_PATH, "train")
    test_paths, test_labels = load_data(cfg.BASE_PATH, "test")

    train_df = pd.DataFrame({
        "path": train_paths,
        "label": train_labels
    })

    test_df = pd.DataFrame({
        "path": test_paths,
        "label": test_labels
    })

    if train_df.empty:
        raise ValueError("No training images were found.")

    if test_df.empty:
        raise ValueError("No test images were found.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_df["path"].values,
        train_df["label"].values,
        test_size=0.20,
        stratify=train_df["label"].values,
        random_state=cfg.RANDOM_STATE
    )

    df_train = pd.DataFrame({
        "path": train_paths,
        "label": train_labels
    })

    df_val = pd.DataFrame({
        "path": val_paths,
        "label": val_labels
    })

    return df_train, df_val, test_df


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.3)
])


def process_image(path, augment=False):
    image = cv2.imread(path)

    if image is None:
        return np.zeros(
            (*cfg.IMG_SIZE, 3),
            dtype=np.float32
        )

    image = cv2.resize(image, cfg.IMG_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augment:
        image = transform(image=image)["image"]

    image = image.astype(np.float32) / 255.0

    return image


label_encoder = LabelEncoder()
label_encoder.fit(cfg.CLASSES)


class OCTGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=True,
        augment=False
    ):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.indices = np.arange(len(self.df))
        self.labels_encoded = label_encoder.transform(
            self.df["label"].values
        )

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(len(self.df) / self.batch_size)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        images = []
        labels = []

        for i in batch_indices:
            row = self.df.iloc[i]

            image = process_image(
                row["path"],
                augment=self.augment
            )

            images.append(image)
            labels.append(self.labels_encoded[i])

        X = np.asarray(images, dtype=np.float32)

        y = tf.keras.utils.to_categorical(
            labels,
            num_classes=len(cfg.CLASSES)
        )

        return X, y


def create_generators():
    df_train, df_val, df_test = create_dataframes()

    train_gen = OCTGenerator(
        df_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_gen = OCTGenerator(
        df_val,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    test_gen = OCTGenerator(
        df_test,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    return train_gen, val_gen, test_gen


def show_generator_samples(generator):
    X_batch, y_batch = generator[0]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )

    for i in range(4):
        axes[0, i].imshow(X_batch[i])
        axes[0, i].set_title(
            cfg.CLASSES[np.argmax(y_batch[i])]
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(X_batch[i + 4])
        axes[1, i].set_title(
            cfg.CLASSES[np.argmax(y_batch[i + 4])]
        )
        axes[1, i].axis("off")

    plt.suptitle("Generator Samples")
    plt.tight_layout()
    plt.show()


def main():
    print("Loading OCT dataset...")

    train_gen, val_gen, test_gen = create_generators()

    print(f"Training batches   : {len(train_gen)}")
    print(f"Validation batches : {len(val_gen)}")
    print(f"Testing batches    : {len(test_gen)}")

    X_batch, y_batch = train_gen[0]

    print(f"Image batch shape  : {X_batch.shape}")
    print(f"Label batch shape  : {y_batch.shape}")

    show_generator_samples(train_gen)


if __name__ == "__main__":
    main()
