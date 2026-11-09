
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# =====================================================
# 1. CAU HINH
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data_split"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

CLASS_NAMES_FILE = PROJECT_DIR / "training" / "class_names.json"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "rice_disease_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "rice_disease_model_best.keras"

HISTORY_PATH = MODEL_DIR / "training_history.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
RANDOM_SEED = 42

# Giới hạn số luồng CPU để hạn chế quá tải.
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# =====================================================
# 2. KIEM TRA DU LIEU
# =====================================================

def check_dataset():
    required_paths = [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        CLASS_NAMES_FILE,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )

    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != 10:
        raise ValueError(
            f"Can 10 nhan lop, nhung tim thay {len(class_names)}"
        )

    for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        actual_classes = sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir()
        )

        if actual_classes != sorted(class_names):
            raise ValueError(
                f"Danh sach lop khong khop tai {split_dir}"
            )

    print("Kiem tra du lieu: THANH CONG")
    print("So lop:", len(class_names))
    print("Thu tu nhan lop:", class_names)

    return class_names


# =====================================================
# 3. DOC ANH
# =====================================================

def load_datasets(class_names):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    # Khong cache toan bo anh de tranh ton RAM.
    train_ds = train_ds.prefetch(1)
    validation_ds = validation_ds.prefetch(1)
    test_ds = test_ds.prefetch(1)

    return train_ds, validation_ds, test_ds


# =====================================================
# 4. XAY DUNG MOBILENETV2
# =====================================================

def build_model(number_of_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Dong bang mang MobileNetV2.
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    # Tang cuong du lieu khi huan luyen.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)

    # Chuan hoa anh theo MobileNetV2.
    x = preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =====================================================
# 5. HUAN LUYEN
# =====================================================

def train_model(model, train_ds, validation_ds):
    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# =====================================================
# 6. DANH GIA TREN TAP TEST
# =====================================================

def evaluate_model(model, test_ds, class_names):
    print("\nBAT DAU DANH GIA TAP TEST")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(
            np.argmax(predictions, axis=1).tolist()
        )

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nKET QUA TEST")
    print("Test loss:", round(float(test_loss), 4))
    print("Test accuracy:", round(float(test_accuracy) * 100, 2), "%")

    print("\nBAO CAO TUNG LOP")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Da luu ket qua:", METRICS_PATH)


# =====================================================
# 7. CHUONG TRINH CHINH
# =====================================================

def main():
    print("=" * 60)
    print("LUAAI - HUAN LUYEN MO HINH NHAN DIEN BENH LUA")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    class_names = check_dataset()

    train_ds, validation_ds, test_ds = load_datasets(
        class_names
    )

    print("\nSO LUONG DU LIEU")
    print("Train: 4692 anh")
    print("Validation: 1007 anh")
    print("Test: 1013 anh")

    print("\nDANG KHOI TAO MOBILENETV2")

    model = build_model(len(class_names))

    model.summary()

    print("\nBAT DAU HUAN LUYEN")

    history = train_model(
        model,
        train_ds,
        validation_ds,
    )

    # Nap lai mo hinh co Validation accuracy tot nhat.
    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    best_model.save(MODEL_PATH)

    # Luu lich su huan luyen.
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(
            history_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # Danh gia tren tap Test.
    evaluate_model(
        best_model,
        test_ds,
        class_names,
    )

    # Luu lai danh sach nhan trong thu muc models.
    with open(
        MODEL_DIR / "class_names.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_names,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("HOAN THANH HUAN LUYEN")
    print("Mo hinh:", MODEL_PATH)
    print("Lich su:", HISTORY_PATH)
    print("Ket qua:", METRICS_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
