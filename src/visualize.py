# 可视化模块：绘制训练曲线、混淆矩阵和第一层权重可视化图。

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


Array = np.ndarray


def plot_training_curves(history: Dict[str, list], out_path: str | Path) -> None:
    """Plot train/val loss and accuracy curves over epochs."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Array,
    class_names: Sequence[str],
    out_path: str | Path,
    normalize: bool = False,
) -> None:
    """Plot and save confusion matrix heatmap."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    matrix = cm.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_first_layer_weights(
    w1: Array,
    input_shape: Tuple[int, int, int],
    out_path: str | Path,
    max_units: int = 25,
) -> None:
    """Visualize first-layer weight vectors as pseudo-RGB images."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    in_dim, hidden_dim = w1.shape
    if np.prod(input_shape) != in_dim:
        raise ValueError("input_shape does not match first-layer weight dimension")

    n = min(max_units, hidden_dim)
    grid = int(math.ceil(math.sqrt(n)))

    fig, axes = plt.subplots(grid, grid, figsize=(2 * grid, 2 * grid))
    axes = np.array(axes).reshape(-1)

    for i in range(grid * grid):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        img = w1[:, i].reshape(input_shape)
        mn, mx = float(np.min(img)), float(np.max(img))
        if mx - mn < 1e-9:
            img_norm = np.zeros_like(img)
        else:
            img_norm = (img - mn) / (mx - mn)
        ax.imshow(img_norm)
        ax.set_title(f"h{i}", fontsize=8)

    fig.suptitle("First-layer Weights", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
