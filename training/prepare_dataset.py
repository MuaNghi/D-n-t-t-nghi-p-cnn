
from pathlib import Path
from collections import Counter
import shutil

from sklearn.model_selection import train_test_split


# Duong dan du an
PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "data_split"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu: {SOURCE_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Thu muc {OUTPUT_DIR} da ton tai. "
            "Khong ghi de du lieu cu. "
            "Hay kiem tra hoac doi ten thu muc nay truoc."
        )

    class_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise ValueError("Khong tim thay thu muc phan loai.")

    # Thu thap anh theo tung lop
    dataset = {}

    for class_dir in class_dirs:
        images = sorted(
            file for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            raise ValueError(
                f"Lop {class_dir.name} co qua it anh: {len(images)}"
            )

        dataset[class_dir.name] = images

    print("BAT DAU CHIA DU LIEU")
    print("=" * 55)

    # Tao thu muc dau ra
    for split in ("train", "validation", "test"):
        for class_name in dataset:
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True
            )

    totals = Counter()

    for class_name, images in dataset.items():

        # Tach train va phan con lai
        train_images, remaining_images = train_test_split(
            images,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        # Tach validation va test
        relative_test_ratio = (
            TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
        )

        validation_images, test_images = train_test_split(
            remaining_images,
            test_size=relative_test_ratio,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        splits = {
            "train": train_images,
            "validation": validation_images,
            "test": test_images,
        }

        for split_name, split_images in splits.items():

            destination = (
                OUTPUT_DIR / split_name / class_name
            )

            for index, source_file in enumerate(split_images):
                new_name = (
                    f"{class_name}_{index:05d}"
                    f"{source_file.suffix.lower()}"
                )

                shutil.copy2(
                    source_file,
                    destination / new_name,
                )

            totals[split_name] += len(split_images)

        print(
            f"{class_name}: "
            f"Train={len(train_images)}, "
            f"Validation={len(validation_images)}, "
            f"Test={len(test_images)}"
        )

    print("=" * 55)
    print("KET QUA CHIA DU LIEU")
    print(f"Train:      {totals['train']} anh")
    print(f"Validation: {totals['validation']} anh")
    print(f"Test:       {totals['test']} anh")
    print(
        f"Tong cong:  "
        f"{sum(totals.values())} anh"
    )

    print(f"\nDu lieu da duoc luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
