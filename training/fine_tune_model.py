
import json
from pathlib import Path

import tensorflow as tf

# =========================================================
# 1. CAU HINH
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data_split" / "train"
VAL_DIR = PROJECT_DIR / "data_split" / "validation"

MODEL_PATH = PROJECT_DIR / "models" / "rice_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

OUTPUT_MODEL = PROJECT_DIR / "models" / "rice_disease_model_finetuned.keras"
BEST_MODEL = PROJECT_DIR / "models" / "rice_disease_model_finetuned_best.keras"
HISTORY_PATH = PROJECT_DIR / "models" / "finetune_history.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 0.00001
UNFREEZE_LAST_N = 30

# =========================================================
# 2. CAU HINH CPU
# =========================================================

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


# =========================================================
# 3. KIEM TRA FILE
# =========================================================

def check_files():
    required_paths = [
        TRAIN_DIR,
        VAL_DIR,
        MODEL_PATH,
        CLASS_NAMES_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )


# =========================================================
# 4. DOC TEN CAC LOP
# =========================================================

def load_class_names():
    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    if len(class_names) != 10:
        raise ValueError(
            "Danh sach nhan phai co 10 lop."
        )

    return class_names


# =========================================================
# 5. TAI DATASET
# =========================================================

def load_datasets(class_names):

    print("\nDANG TAI TAP TRAIN...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    print("\nDANG TAI TAP VALIDATION...")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    return train_ds, val_ds


# =========================================================
# 6. MO KHOA 30 LOP CUOI MOBILENETV2
# =========================================================

def configure_fine_tuning(model):

    base_model = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    base_model.trainable = True

    # Dong bang cac lop phia truoc
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    # Luon dong bang Batch Normalization
    for layer in base_model.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    trainable_layers = [
        layer.name
        for layer in base_model.layers
        if layer.trainable
    ]

    print("\n" + "=" * 60)
    print("CAU HINH FINE-TUNING")
    print("=" * 60)

    print("Tong so lop MobileNetV2:", len(base_model.layers))
    print("So lop duoc mo khoa:", len(trainable_layers))
    print("So lop duoc dong bang:",
          len(base_model.layers) - len(trainable_layers))

    print("\nCac lop duoc mo khoa:")

    for name in trainable_layers:
        print("-", name)

    print("\nLearning Rate:", LEARNING_RATE)
    print("Batch Size:", BATCH_SIZE)
    print("So Epoch toi da:", EPOCHS)

    # Compile lai sau khi thay doi trainable
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================================================
# 7. HUAN LUYEN
# =========================================================

def train(model, train_ds, val_ds):

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=0.0000001,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print("BAT DAU FINE-TUNING MOBILENETV2")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Luu mo hinh voi trong so tot nhat
    model.save(OUTPUT_MODEL)

    # Luu lich su huan luyen
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("FINE-TUNING HOAN TAT")
    print("=" * 60)

    print("Mo hinh moi:", OUTPUT_MODEL)
    print("Mo hinh tot nhat:", BEST_MODEL)
    print("Lich su huan luyen:", HISTORY_PATH)

    if "val_accuracy" in history.history:
        best_accuracy = max(
            history.history["val_accuracy"]
        )

        print(
            "Validation Accuracy tot nhat:",
            f"{best_accuracy * 100:.2f}%"
        )


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("LUAAI - FINE-TUNING MOBILENETV2")
    print("=" * 60)

    check_files()

    class_names = load_class_names()

    print("\nSo lop phan loai:", len(class_names))

    print("\nDANG TAI MO HINH BASELINE...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Tai mo hinh thanh cong.")

    train_ds, val_ds = load_datasets(class_names)

    model = configure_fine_tuning(model)

    train(model, train_ds, val_ds)


if __name__ == "__main__":
    main()
