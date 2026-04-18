# 测试评估模块：计算分类指标并生成混淆矩阵所需结果。

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.model import MLP


Array = np.ndarray


def softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(logits: Array, y: Array) -> Tuple[float, Array]:
    """Return (loss, dlogits) for CE with integer class targets."""
    probs = softmax(logits)
    n = y.shape[0]
    eps = 1e-12
    loss = -np.mean(np.log(probs[np.arange(n), y] + eps))

    dlogits = probs
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    return float(loss), dlogits


def evaluate_model(
    model: MLP,
    x: Array,
    y: Array,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Compute average CE loss and accuracy on a full split."""
    n = x.shape[0]
    total_loss = 0.0
    total_correct = 0

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        logits, _ = model.forward(x[s:e])
        loss, _ = cross_entropy_loss(logits, y[s:e])
        preds = np.argmax(logits, axis=1)

        total_loss += loss * (e - s)
        total_correct += int(np.sum(preds == y[s:e]))

    return {
        "loss": total_loss / n,
        "accuracy": total_correct / n,
    }


def predict(model: MLP, x: Array, batch_size: int = 1024) -> Array:
    """Predict class ids."""
    return model.predict(x, batch_size=batch_size)


def confusion_and_report(
    y_true: Array,
    y_pred: Array,
    class_names: list[str],
) -> Dict[str, object]:
    """Return confusion matrix and classification report."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "confusion_matrix": cm,
        "classification_report": report,
    }
