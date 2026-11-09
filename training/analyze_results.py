
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

history_path = PROJECT_DIR / "models" / "training_history.json"
metrics_path = PROJECT_DIR / "models" / "test_metrics.json"

with open(history_path, "r", encoding="utf-8") as file:
    history = json.load(file)

with open(metrics_path, "r", encoding="utf-8") as file:
    metrics = json.load(file)

print("=" * 75)
print("LICH SU HUAN LUYEN")
print("=" * 75)

print("So Epoch:", len(history["loss"]))

print(
    f"{'Epoch':<8}"
    f"{'Train Acc':<14}"
    f"{'Val Acc':<14}"
    f"{'Train Loss':<14}"
    f"{'Val Loss':<14}"
)

for i in range(len(history["loss"])):
    print(
        f"{i + 1:<8}"
        f"{history['accuracy'][i]:<14.4f}"
        f"{history['val_accuracy'][i]:<14.4f}"
        f"{history['loss'][i]:<14.4f}"
        f"{history['val_loss'][i]:<14.4f}"
    )

print("\n" + "=" * 75)
print("MA TRAN NHAM LAN")
print("=" * 75)

class_names = metrics["class_names"]
matrix = metrics["confusion_matrix"]

print("Thu tu cac lop:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nMa tran:")
print("Actual / Predicted")

print(" " * 24, end="")
for i in range(len(class_names)):
    print(f"{i:5}", end="")
print()

for i, row in enumerate(matrix):
    print(f"{class_names[i]:<24}", end="")
    for value in row:
        print(f"{value:5}", end="")
    print()

print("\nHOAN THANH PHAN TICH")
