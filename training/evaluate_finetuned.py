
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data_split" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "rice_disease_model_finetuned_best.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

REPORT_DIR = PROJECT_DIR / "reports"
METRICS_PATH = REPORT_DIR / "finetuned_test_metrics.json"
REPORT_PATH = REPORT_DIR / "finetuned_classification_report.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


def main():
    print("=" * 60)
    print("LUAAI - DANH GIA MO HINH FINE-TUNING")
    print("=" * 60)

    for path in [TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay: {path}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print("\nDANG TAI MO HINH...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )
    print("Tai mo hinh thanh cong.")

    print("\nDANG TAI TAP TEST...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    print("\nDANG DU DOAN...")
    predictions = model.predict(test_ds, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    metrics = {
        "model": "MobileNetV2 Fine-tuned",
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("KET QUA DANH GIA TAP TEST")
    print("=" * 60)
    print("Tong anh:", len(y_true))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nBAO CAO THEO TUNG LOP")
    print(report)

    print("CONFUSION MATRIX")
    print(matrix)

    print("\nFILE DA LUU:")
    print(METRICS_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
