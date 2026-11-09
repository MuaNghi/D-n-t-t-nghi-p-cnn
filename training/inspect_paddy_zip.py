from pathlib import Path
from zipfile import ZipFile
from collections import Counter

zip_path = Path(
    r"D:\Bộ dữ liệu Lúa AI\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Field Pest Image Dataset for Deep Learning-B\Paddy Final.zip"
)

if not zip_path.exists():
    print("KHONG TIM THAY FILE ZIP!")
    raise SystemExit(1)

print("FILE ZIP:", zip_path)
print("DUNG LUONG:", round(zip_path.stat().st_size / 1024 / 1024, 2), "MB")

with ZipFile(zip_path, "r") as z:
    files = [item for item in z.infolist() if not item.is_dir()]

    extensions = Counter(
        Path(item.filename).suffix.lower() or "[khong co duoi]"
        for item in files
    )

    print("\nTONG SO FILE:", len(files))
    print("TONG DUNG LUONG SAU GIAI NEN:",
          round(sum(item.file_size for item in files) / 1024 / 1024, 2), "MB")

    print("\nTHONG KE DINH DANG:")
    for ext, count in extensions.most_common():
        print(f"{ext}: {count:,} file")

    print("\n100 DUONG DAN DAU TIEN:")
    for item in files[:100]:
        print(item.filename)

    print("\nKIEM TRA ZIP:")
    print("ZIP TO HOP LE" if z.testzip() is None else "ZIP CO FILE LOI")
