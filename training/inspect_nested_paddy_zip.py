
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from collections import Counter

outer_path = Path(
    r"D:\Bộ dữ liệu Lúa AI\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Final.zip"
)

image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

total_images = 0

with ZipFile(outer_path, "r") as outer:
    nested_files = [
        item for item in outer.infolist()
        if item.filename.lower().endswith(".zip")
    ]

    for index, item in enumerate(nested_files, start=1):
        print("\n" + "=" * 75)
        print(f"BO DU LIEU {index}: {item.filename}")
        print("=" * 75)

        nested_bytes = outer.read(item.filename)

        with ZipFile(BytesIO(nested_bytes), "r") as inner:
            class_counts = Counter()
            invalid_paths = 0

            for file in inner.infolist():
                if file.is_dir():
                    continue

                extension = Path(file.filename).suffix.lower()

                if extension not in image_extensions:
                    continue

                parts = Path(file.filename).parts

                if len(parts) < 3:
                    invalid_paths += 1
                    continue

                class_name = parts[1]
                class_counts[class_name] += 1

            dataset_total = sum(class_counts.values())
            total_images += dataset_total

            print("\nSO LUONG LOP:", len(class_counts))
            print("TONG SO ANH:", f"{dataset_total:,}")
            print("DUONG DAN KHONG XAC DINH NHAN:", invalid_paths)

            print("\nCHI TIET SO ANH THEO LOP:")
            print("-" * 55)

            for class_name, count in sorted(class_counts.items()):
                print(f"{class_name:30} | {count:>7,} anh")

            print("-" * 55)

            result = inner.testzip()
            print(
                "\nKIEM TRA ZIP:",
                "HOP LE" if result is None else f"CO FILE LOI: {result}"
            )

print("\n" + "=" * 75)
print("TONG KET CA 3 BO DU LIEU")
print("=" * 75)
print("TONG SO ANH:", f"{total_images:,}")
