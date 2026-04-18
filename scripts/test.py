# 测试入口脚本：加载已保存模型并在测试集上执行评估。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import collect_misclassified, plot_misclassified_grid, save_misclassified_json
from src.data import load_eurosat_splits
from src.evaluate import confusion_and_report, evaluate_model, predict
from src.trainer import build_model_from_checkpoint
from src.utils import ensure_dir, save_json
from src.visualize import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved MLP checkpoint on EuroSAT test split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="runs/test_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    model, meta = build_model_from_checkpoint(args.checkpoint)

    # Prefer split settings embedded in checkpoint if available.
    split_meta = meta.get("split", {}) if isinstance(meta, dict) else {}
    val_ratio = float(split_meta.get("val_ratio", args.val_ratio))
    test_ratio = float(split_meta.get("test_ratio", args.test_ratio))
    split_seed = int(split_meta.get("seed", args.seed))
    image_size = int(split_meta.get("image_size", args.image_size))

    dataset = load_eurosat_splits(
        data_dir=args.data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
        image_size=image_size,
        max_per_class=args.max_per_class,
    )

    x_test, y_test = dataset["test"]["X"], dataset["test"]["y"]
    class_names = dataset["class_names"]

    metrics = evaluate_model(model, x_test, y_test)
    y_pred = predict(model, x_test)
    details = confusion_and_report(y_test, y_pred, class_names)
    cm = details["confusion_matrix"]

    plot_confusion_matrix(cm, class_names, Path(out_dir) / "confusion_matrix_test.png")
    plot_confusion_matrix(cm, class_names, Path(out_dir) / "confusion_matrix_test_norm.png", normalize=True)

    mis = collect_misclassified(
        y_true=y_test,
        y_pred=y_pred,
        paths=dataset["test"]["paths"],
        class_names=class_names,
        max_items=36,
    )
    plot_misclassified_grid(mis, Path(out_dir) / "misclassified_test.png")
    save_misclassified_json(mis, Path(out_dir) / "misclassified_test.json")

    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "test_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": details["classification_report"],
        },
        Path(out_dir) / "test_summary.json",
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
