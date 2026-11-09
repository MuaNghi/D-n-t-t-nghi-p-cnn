
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from PIL import Image, ImageDraw
import random
import math

# Duong dan bo ZIP goc
OUTER_ZIP = Path(
    r"D:\Bộ dữ liệu Lúa AI\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Final.zip"
)

# Thu muc luu ket qua
OUTPUT_DIR = Path("reports/pest_dataset/paddy_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TARGET_CLASSES = [
    "Brown Planthopper",
    "Green Leafhoppers",
    "Leaf Folders",
    "Rice Bug",
    "Stem Borer",
    "Whorl-Maggot",
]

SAMPLES_PER_CLASS = 20
COLUMNS = 5
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 140
LABEL_HEIGHT = 25

random.seed(42)


def create_contact_sheet(zip_file, class_name):
    candidates = []

    for item in zip_file.infolist():
        if item.is_dir():
            continue

        if Path(item.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = Path(item.filename).parts

        if len(parts) >= 3 and parts[1] == class_name:
            candidates.append(item)

    random.shuffle(candidates)
    selected = candidates[:SAMPLES_PER_CLASS]

    if not selected:
        print("KHONG TIM THAY ANH:", class_name)
        return

    rows = math.ceil(len(selected) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * IMAGE_WIDTH,
            rows * (IMAGE_HEIGHT + LABEL_HEIGHT) + 35
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 8),
        f"{class_name} | {len(selected)}/{len(candidates)} anh",
        fill="black"
    )

    valid_count = 0

    for index, item in enumerate(selected):
        try:
            image_bytes = zip_file.read(item.filename)

            with Image.open(BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                image.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))

            column = index % COLUMNS
            row = index // COLUMNS

            x = column * IMAGE_WIDTH
            y = 35 + row * (IMAGE_HEIGHT + LABEL_HEIGHT)

            background = Image.new(
                "RGB",
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                "#eeeeee"
            )

            offset = (
                (IMAGE_WIDTH - image.width) // 2,
                (IMAGE_HEIGHT - image.height) // 2
            )

            background.paste(image, offset)
            sheet.paste(background, (x, y))

            draw.text(
                (x + 3, y + IMAGE_HEIGHT + 3),
                f"Anh {index + 1}",
                fill="black"
            )

            valid_count += 1

        except Exception as error:
            print("LOI DOC ANH:", item.filename, error)

    output_path = OUTPUT_DIR / f"{class_name.replace(' ', '_')}.jpg"

    sheet.save(output_path, quality=92)

    print("Lop:", class_name)
    print("Tong anh trong ZIP:", len(candidates))
    print("Anh dua vao contact sheet:", valid_count)
    print("Da luu:", output_path.resolve())


def main():
    if not OUTER_ZIP.exists():
        print("KHONG TIM THAY FILE ZIP:", OUTER_ZIP)
        return

    with ZipFile(OUTER_ZIP, "r") as outer:
        nested_item = next(
            (
                item for item in outer.infolist()
                if "raw data-" in item.filename.lower()
                and item.filename.lower().endswith(".zip")
            ),
            None
        )

        if nested_item is None:
            print("KHONG TIM THAY raw data ZIP!")
            return

        print("Dang doc:", nested_item.filename)

        nested_bytes = outer.read(nested_item.filename)

        with ZipFile(BytesIO(nested_bytes), "r") as inner:
            for class_name in TARGET_CLASSES:
                print("\n" + "=" * 55)
                create_contact_sheet(inner, class_name)

    print("\nHOAN THANH TAO 6 CONTACT SHEET!")


if __name__ == "__main__":
    main()
