
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORT_CSV = (
    PROJECT_DIR
    / "reports"
    / "misclassified_20261108_074243"
    / "misclassified_report.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "leaf_blast_brown_spot_contact_sheet.png"
)

TARGET_PAIRS = {
    ("leaf_blast", "brown_spot"),
    ("brown_spot", "leaf_blast"),
}

COLUMNS = 4
CARD_WIDTH = 360
CARD_HEIGHT = 330
IMAGE_WIDTH = 330
IMAGE_HEIGHT = 245
PADDING = 15


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    if not REPORT_CSV.exists():
        raise FileNotFoundError(
            f"Khong tim thay bao cao: {REPORT_CSV}"
        )

    selected_images = []

    with open(
        REPORT_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            pair = (
                row["Actual class"],
                row["Predicted class"],
            )

            if pair in TARGET_PAIRS:
                selected_images.append(row)

    if not selected_images:
        print("Khong tim thay anh thuoc hai nhom can kiem tra.")
        return

    rows = (
        len(selected_images) + COLUMNS - 1
    ) // COLUMNS

    sheet_width = COLUMNS * CARD_WIDTH
    sheet_height = rows * CARD_HEIGHT

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    title_font = load_font(17)
    info_font = load_font(14)

    for index, item in enumerate(selected_images):
        actual = item["Actual class"]
        predicted = item["Predicted class"]
        confidence = float(item["Prediction confidence"])

        image_path = Path(item["Export path"])

        if not image_path.exists():
            print(f"Bo qua anh khong ton tai: {image_path}")
            continue

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")

        image.thumbnail(
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        column = index % COLUMNS
        row = index // COLUMNS

        x = column * CARD_WIDTH
        y = row * CARD_HEIGHT

        image_x = x + (CARD_WIDTH - image.width) // 2
        image_y = y + 10

        sheet.paste(image, (image_x, image_y))

        draw.text(
            (x + 15, y + 265),
            f"Actual: {actual}",
            fill="black",
            font=title_font
        )

        draw.text(
            (x + 15, y + 290),
            f"Predicted: {predicted}",
            fill="red",
            font=info_font
        )

        draw.text(
            (x + 15, y + 310),
            f"Confidence: {confidence:.2%}",
            fill="blue",
            font=info_font
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(OUTPUT_PATH)

    print("=" * 55)
    print("DA TAO ANH TONG HOP THANH CONG")
    print("=" * 55)
    print("So anh tim thay:", len(selected_images))
    print("So cot:", COLUMNS)
    print("Duong dan anh:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
