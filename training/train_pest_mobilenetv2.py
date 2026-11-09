import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# 1. CAU HINH
# =========================================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 8

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "pest_dataset" / "split"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "pest"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("=" * 60)
print("LUAAI - PEST RECOGNITION - MOBILENETV2")
print("=" * 60)
print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

# =========================================================
# 2. KIEM TRA DU LIEU
# =========================================================

for split in ["train", "validation", "test"]:
    folder = DATA_DIR / split
    if not folder.exists():
        raise FileNotFoundError(f"Thieu thu muc: {folder}")

    if not any(folder.iterdir()):
        raise RuntimeError(f"Thu muc rong: {folder}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes != 6:
    raise RuntimeError(f"Can 6 lop, nhung tim thay {num_classes} lop.")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\n===== DANH SACH LOP =====")
for index, name in enumerate(class_names):
    print(index, ":", name)

with open(REPORT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

# =========================================================
# 3. TINH CLASS WEIGHT
# =========================================================

train_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("\n===== CLASS WEIGHTS =====")
for index, name in enumerate(class_names):
    print(f"{name}: {class_weights[index]:.4f}")

# Giam tai bo nho va CPU
options = tf.data.Options()
options.threading.private_threadpool_size = 4
options.threading.max_intra_op_parallelism = 1

train_ds = train_ds.with_options(options).prefetch(1)
val_ds = val_ds.with_options(options).prefetch(1)
test_ds = test_ds.with_options(options).prefetch(1)

# =========================================================
# 4. XAY DUNG MOBILENETV2
# =========================================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10)
], name="data_augmentation")

backbone = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

backbone.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="pest_predictions"
)(x)

model = tf.keras.Model(inputs, outputs, name="pest_mobilenetv2")

print("\nTong tham so:", f"{model.count_params():,}")

# =========================================================
# 5. GIAI DOAN 1 - HUAN LUYEN PHAN PHAN LOAI
# =========================================================

phase1_path = MODEL_DIR / "pest_mobilenetv2_phase1_best.keras"

def make_callbacks(path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 60)
print("GIAI DOAN 1 - FROZEN BACKBONE")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(phase1_path)
)

# =========================================================
# 6. GIAI DOAN 2 - FINE TUNING
# =========================================================

print("\n" + "=" * 60)
print("GIAI DOAN 2 - FINE TUNING 30 LOP CUOI")
print("=" * 60)

backbone.trainable = True

for layer in backbone.layers[:-30]:
    layer.trainable = False

for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

phase2_path = MODEL_DIR / "pest_mobilenetv2_phase2_best.keras"

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(phase2_path)
)

# =========================================================
# 7. CHON CHECKPOINT TOT NHAT THEO VALIDATION
# =========================================================

print("\n===== SO SANH CHECKPOINT =====")

candidates = [
    ("phase1", phase1_path),
    ("phase2", phase2_path)
]

best_name = None
best_path = None
best_accuracy = -1.0

for name, path in candidates:
    if not path.exists():
        continue

    candidate = tf.keras.models.load_model(path)
    result = candidate.evaluate(val_ds, verbose=0)
    accuracy = float(result[1])

    print(f"{name}: validation accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name
        best_path = path

if best_path is None:
    raise RuntimeError("Khong tim thay checkpoint hop le.")

best_model_path = MODEL_DIR / "pest_mobilenetv2_best.keras"
shutil.copy2(best_path, best_model_path)

model = tf.keras.models.load_model(best_model_path)

print("\nCheckpoint duoc chon:", best_name)
print("Validation accuracy:", best_accuracy)
print("Model:", best_model_path)

# =========================================================
# 8. DANH GIA TREN TEST
# =========================================================

print("\n" + "=" * 60)
print("DANH GIA CUOI CUNG TREN TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    digits=4,
    zero_division=0
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

print("\n===== CLASSIFICATION REPORT =====")
print(report)

print("Test loss:", float(test_loss))
print("Test accuracy:", float(test_accuracy))

with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
    f.write(f"\nTest loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_accuracy:.6f}\n")

np.savetxt(
    REPORT_DIR / "confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d"
)

metrics = {
    "model": "MobileNetV2",
    "classes": class_names,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "best_checkpoint": best_name
}

with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n===== HOAN TAT HUAN LUYEN =====")
print("Model:", best_model_path)
print("Bao cao:", REPORT_DIR)
