
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATA_DIR.exists():
    print(f"KHONG TIM THAY THU MUC: {DATA_DIR}")
else:
    total_images = 0

    print("KIEM TRA DU LIEU ANH LA LUA")
    print("-" * 45)

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += count
        print(f"{class_dir.name}: {count} anh")

    print("-" * 45)
    print(f"TONG SO ANH: {total_images}")
