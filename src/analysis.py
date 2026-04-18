# 错例分析模块：收集并展示预测错误样本及其预测信息。

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils import ensure_dir, save_json


Array = np.ndarray


def collect_misclassified(
    y_true: Array,
    y_pred: Array,
    paths: Sequence[str],
    class_names: Sequence[str],
    max_items: int = 25,
) -> List[Dict[str, object]]:
    """Collect misclassified samples for error analysis."""
    wrong = np.where(y_true != y_pred)[0]
    records: List[Dict[str, object]] = []

    for idx in wrong[:max_items]:
        records.append(
            {
                "index": int(idx),
                "path": str(paths[idx]),
                "true_id": int(y_true[idx]),
                "pred_id": int(y_pred[idx]),
                "true_name": class_names[int(y_true[idx])],
                "pred_name": class_names[int(y_pred[idx])],
            }
        )
    return records


def save_misclassified_json(records: List[Dict[str, object]], out_path: str | Path) -> None:
    """Save misclassified record list to JSON file."""
    save_json({"misclassified": records, "count": len(records)}, out_path)


def plot_misclassified_grid(
    records: List[Dict[str, object]],
    out_path: str | Path,
    max_items: int = 25,
) -> None:
    """Plot a gallery of misclassified images with true/pred labels."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if not records:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified samples", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    records = records[:max_items]
    n = len(records)
    grid = int(math.ceil(math.sqrt(n)))

    fig, axes = plt.subplots(grid, grid, figsize=(2.6 * grid, 2.6 * grid))
    axes = np.array(axes).reshape(-1)

    for i in range(grid * grid):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        item = records[i]
        with Image.open(item["path"]) as img:
            ax.imshow(img.convert("RGB"))

        ax.set_title(
            f"T:{item['true_name']}\nP:{item['pred_name']}",
            fontsize=8,
            color="red",
        )

    fig.suptitle("Misclassified Examples", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
