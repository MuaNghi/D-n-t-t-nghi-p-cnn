
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from PIL import Image, ImageDraw
import random
import math

OUTER_ZIP = Path(
    r"D:\Bộ dữ liệu Lúa AI\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Field Pest ImageDataset for Deep Learning-B\Paddy Final.zip"
)

OUTPUT_DIR = Path("reports/pest_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TARGET_CLASSES = [
    "Brown_Planthopper",
    "Green_Leafhoppers",
    "LEAF_FOLDERS",
    "Rice_Bug",
    "Stemz_Borer",
    "Whorl_Maggot",
]

SAMPLES_PER_CLASS = 12
IMAGE_SIZE = (180, 140)
COLUMNS = 4
ROWS_PER_CLASS = math.ceil(SAMPLES_PER_CLASS / COLUMNS)
LABEL_HEIGHT = 35
CLASS_GAP = 25

random.seed(42)


def load_samples(zip_file, target_class):
    candidates = []

    for item in zip_file.infolist():
        if item.is_dir():
            continue

        if Path(item.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = Path(item.filename).parts

        if len(parts) >= 3 and parts[1] == target_class:
            candidates.append(item)

    random.shuffle(candidates)
    selected = candidates[:SAMPLES_PER_CLASS]

    samples = []

    for item in selected:
        try:
            image_bytes = zip_file.read(item.filename)

            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                image.thumbnail(IMAGE_SIZE)
                samples.append((image.copy(), item.filename))

        except Exception as error:
            print("Loi anh:", item.filename, error)

    return samples, len(candidates)


with ZipFile(OUTER_ZIP, "r") as outer:
    nested_item = next(
        item for item in outer.infolist()
        if "augmented_data" in item.filename
        and item.filename.lower().endswith(".zip")
    )

    nested_bytes = outer.read(nested_item.filename)

    with ZipFile(BytesIO(nested_bytes), "r") as inner:
        for class_name in TARGET_CLASSES:
            print("\nDang xu ly:", class_name)

            samples, total = load_samples(inner, class_name)

            if not samples:
                print("KHONG TIM THAY ANH!")
                continue

            width = COLUMNS * IMAGE_SIZE[0]
            height = (
                ROWS_PER_CLASS * (IMAGE_SIZE[1] + LABEL_HEIGHT)
                + CLASS_GAP
            )

            sheet = Image.new(
                "RGB",
                (width, height),
                "white"
            )

            draw = ImageDraw.Draw(sheet)

            draw.text(
                (10, 5),
                f"{class_name} | Tong anh: {total}",
                fill="black"
            )

            for index, (image, filename) in enumerate(samples):
                row = index // COLUMNS
                column = index % COLUMNS

                x = column * IMAGE_SIZE[0]
                y = 30 + row * (IMAGE_SIZE[1] + LABEL_HEIGHT)

                background = Image.new(
                    "RGB",
                    IMAGE_SIZE,
                    "#eeeeee"
                )

                offset = (
                    (IMAGE_SIZE[0] - image.width) // 2,
                    (IMAGE_SIZE[1] - image.height) // 2
                )

                background.paste(image, offset)
                sheet.paste(background, (x, y))

                short_name = Path(filename).name[:22]

                draw.text(
                    (x + 3, y + IMAGE_SIZE[1] + 3),
                    short_name,
                    fill="black"
                )

            output_path = OUTPUT_DIR / f"{class_name}_contact_sheet.jpg"

            sheet.save(
                output_path,
                quality=90
            )

            print("So anh hien thi:", len(samples))
            print("Da luu:", output_path.resolve())

print("\nHOAN THANH TAO CONTACT SHEET!")
