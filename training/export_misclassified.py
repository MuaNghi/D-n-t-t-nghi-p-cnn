
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

# =====================================================
# 1. DUONG DAN
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data_split" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "rice_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

REPORTS_DIR = PROJECT_DIR / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


# =====================================================
# 2. KIEM TRA FILE
# =====================================================

def check_files():
    required_files = [
        TEST_DIR,
        MODEL_PATH,
        CLASS_NAMES_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )


# =====================================================
# 3. DOC DANH SACH NHAN
# =====================================================

def load_class_names():
    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_names = json.load(file)

    if len(class_names) != 10:
        raise ValueError("Danh sach nhan khong co 10 lop.")

    return class_names


# =====================================================
# 4. DOC TAP TEST
# =====================================================

def load_test_dataset(class_names):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    image_paths = test_ds.file_paths

    return test_ds, image_paths


# =====================================================
# 5. DU DOAN VA XUAT ANH NHAM
# =====================================================

def export_misclassified(model, test_ds, image_paths, class_names):

    print("\nDANG DU DOAN TREN TAP TEST...")

    predictions = model.predict(
        test_ds,
        verbose=1,
    )

    y_pred = np.argmax(predictions, axis=1)

    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    if not (
        len(image_paths) == len(y_true) == len(y_pred)
    ):
        raise ValueError(
            "So luong anh, nhan va ket qua du doan khong khop."
        )

    # Tao thu muc rieng cho moi lan xuat.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = REPORTS_DIR / f"misclassified_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "misclassified_report.csv"

    wrong_count = 0
    wrong_by_class = {name: 0 for name in class_names}

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "Image",
            "Actual class",
            "Predicted class",
            "Prediction confidence",
            "Actual class probability",
            "Source path",
            "Export path",
        ])

        for index in range(len(y_true)):

            actual_index = int(y_true[index])
            predicted_index = int(y_pred[index])

            # Bo qua anh duoc nhan dien dung.
            if actual_index == predicted_index:
                continue

            actual_class = class_names[actual_index]
            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[index][predicted_index]
            )

            actual_probability = float(
                predictions[index][actual_index]
            )

            source_path = Path(image_paths[index])

            # Sap xep theo nhan thuc te va nhan du doan.
            destination_dir = (
                output_dir
                / f"{actual_class}__predicted_as__{predicted_class}"
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            new_filename = (
                f"{index + 1:04d}_{source_path.name}"
            )

            destination_path = destination_dir / new_filename

            shutil.copy2(
                source_path,
                destination_path,
            )

            writer.writerow([
                source_path.name,
                actual_class,
                predicted_class,
                round(confidence, 6),
                round(actual_probability, 6),
                str(source_path),
                str(destination_path),
            ])

            wrong_count += 1
            wrong_by_class[actual_class] += 1

    # In ket qua.
    print("\n" + "=" * 60)
    print("KET QUA XUAT ANH NHAN DIEN SAI")
    print("=" * 60)

    print("Tong anh Test:", len(y_true))
    print("Nhan dien sai:", wrong_count)
    print("Nhan dien dung:", len(y_true) - wrong_count)

    print("\nSO ANH NHAN DIEN SAI THEO LOP")

    for class_name, count in wrong_by_class.items():
        print(f"{class_name}: {count} anh")

    print("\nTHU MUC ANH:")
    print(output_dir)

    print("\nFILE BAO CAO:")
    print(csv_path)

    print("\nHOAN THANH!")


# =====================================================
# 6. CHUONG TRINH CHINH
# =====================================================

def main():
    print("=" * 60)
    print("LUAAI - XUAT ANH NHAN DIEN SAI")
    print("=" * 60)

    check_files()

    class_names = load_class_names()

    print("\nDANG TAI MO HINH...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Tai mo hinh thanh cong.")

    test_ds, image_paths = load_test_dataset(class_names)

    export_misclassified(
        model,
        test_ds,
        image_paths,
        class_names,
    )


if __name__ == "__main__":
    main()
