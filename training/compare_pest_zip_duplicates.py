
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
import hashlib

outer_path = Path(
    r"D:\Bộ dữ liệu Lúa AI\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Final.zip"
)

image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

datasets = {}

with ZipFile(outer_path, "r") as outer:
    nested_files = [
        item for item in outer.infolist()
        if item.filename.lower().endswith(".zip")
    ]

    for item in nested_files:
        dataset_name = Path(item.filename).name.split("-2025")[0]

        print(f"\nDang kiem tra: {dataset_name}")

        hashes = {}

        nested_bytes = outer.read(item.filename)

        with ZipFile(BytesIO(nested_bytes), "r") as inner:
            for file in inner.infolist():
                if file.is_dir():
                    continue

                if Path(file.filename).suffix.lower() not in image_extensions:
                    continue

                parts = Path(file.filename).parts

                if len(parts) < 3:
                    continue

                class_name = parts[1]
                image_bytes = inner.read(file.filename)

                image_hash = hashlib.sha256(image_bytes).hexdigest()

                hashes.setdefault(image_hash, []).append(class_name)

        datasets[dataset_name] = hashes

        duplicate_inside = sum(
            len(paths) - 1
            for paths in hashes.values()
            if len(paths) > 1
        )

        print("So anh:", sum(len(v) for v in hashes.values()))
        print("So hash duy nhat:", len(hashes))
        print("Anh trung lap trong cung ZIP:", duplicate_inside)

print("\n" + "=" * 70)
print("SO SANH ANH TRUNG GIUA CAC ZIP")
print("=" * 70)

names = list(datasets.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name_a = names[i]
        name_b = names[j]

        hashes_a = datasets[name_a]
        hashes_b = datasets[name_b]

        common = set(hashes_a) & set(hashes_b)

        same_class = 0
        different_class = 0

        for image_hash in common:
            classes_a = set(hashes_a[image_hash])
            classes_b = set(hashes_b[image_hash])

            if classes_a & classes_b:
                same_class += 1
            else:
                different_class += 1

        print(f"\n{name_a} <-> {name_b}")
        print("Anh trung khop SHA-256:", len(common))
        print("Trung cung nhan:", same_class)
        print("Trung khac nhan:", different_class)

print("\nHOAN THANH KIEM TRA!")
