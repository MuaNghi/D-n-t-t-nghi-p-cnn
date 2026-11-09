
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

BASELINE_PATH = PROJECT_DIR / "models" / "test_metrics.json"
FINETUNED_PATH = PROJECT_DIR / "reports" / "finetuned_test_metrics.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    baseline = load_json(BASELINE_PATH)
    finetuned = load_json(FINETUNED_PATH)

    baseline_report = baseline["classification_report"]
    finetuned_report = finetuned["classification_report"]

    class_names = finetuned["class_names"]

    baseline_accuracy = baseline["test_accuracy"]
    finetuned_accuracy = finetuned["accuracy"]

    print("=" * 75)
    print("LUAAI - SO SANH BASELINE VA FINE-TUNING")
    print("=" * 75)

    print(f"\nBaseline Accuracy  : {baseline_accuracy * 100:.2f}%")
    print(f"Fine-tuned Accuracy: {finetuned_accuracy * 100:.2f}%")

    improvement = (finetuned_accuracy - baseline_accuracy) * 100

    print(f"Cai thien          : {improvement:+.2f} diem %")

    print("\nSO SANH THEO TUNG LOP")
    print("-" * 75)

    print(
        f"{'Class':25}"
        f"{'Baseline F1':>13}"
        f"{'Fine-tuned F1':>15}"
        f"{'Thay doi':>12}"
    )

    for name in class_names:
        old_f1 = baseline_report[name]["f1-score"]
        new_f1 = finetuned_report[name]["f1-score"]

        difference = (new_f1 - old_f1) * 100

        print(
            f"{name:25}"
            f"{old_f1 * 100:12.2f}%"
            f"{new_f1 * 100:14.2f}%"
            f"{difference:+11.2f}"
        )

    print("\nHOAN TAT SO SANH.")


if __name__ == "__main__":
    main()
