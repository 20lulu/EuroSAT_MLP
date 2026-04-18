# 训练流程模块：组织批训练循环、SGD 更新、学习率衰减与模型保存逻辑。

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.data import iterate_minibatches
from src.evaluate import cross_entropy_loss, evaluate_model
from src.model import MLP, MLPConfig
from src.utils import ensure_dir


Array = np.ndarray


def save_checkpoint(model: MLP, path: str | Path, meta: Dict[str, object]) -> None:
    """Save model parameters and JSON metadata into a single .npz file."""
    path = Path(path)
    ensure_dir(path.parent)
    payload = model.state_dict()
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez(path, **payload)


def load_checkpoint(path: str | Path) -> Tuple[Dict[str, Array], Dict[str, object]]:
    """Load (state_dict, meta) from a checkpoint .npz file."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        state = {k: data[k] for k in data.files if k != "meta_json"}
        meta_json = data["meta_json"].item() if "meta_json" in data.files else "{}"
    meta = json.loads(meta_json)
    return state, meta


def build_model_from_checkpoint(path: str | Path) -> Tuple[MLP, Dict[str, object]]:
    """Construct and load an MLP instance from checkpoint."""
    state, meta = load_checkpoint(path)

    if "model" in meta:
        model_meta = meta["model"]
        input_dim = int(model_meta["input_dim"])
        hidden_dim = int(model_meta["hidden_dim"])
        num_classes = int(model_meta["num_classes"])
        activation = str(model_meta["activation"])
    else:
        # Fallback: infer from state shape if old meta format is missing.
        input_dim = int(state["W1"].shape[0])
        hidden_dim = int(state["W1"].shape[1])
        num_classes = int(state["W2"].shape[1])
        activation = "relu"

    config = MLPConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
        seed=0,
    )
    model = MLP(config)
    model.load_state_dict(state)
    return model, meta


def train_model(
    model: MLP,
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_decay: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    checkpoint_meta: Optional[Dict[str, object]] = None,
    seed: int = 42,
    early_stop_patience: Optional[int] = 10,
) -> Dict[str, object]:
    """Train model with SGD and save best checkpoint by validation accuracy."""
    rng = np.random.default_rng(seed)
    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        current_lr = learning_rate * (lr_decay ** (epoch - 1))
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        iterator = iterate_minibatches(
            x_train,
            y_train,
            batch_size=batch_size,
            rng=rng,
            shuffle=True,
        )

        for xb, yb in tqdm(iterator, total=(len(x_train) + batch_size - 1) // batch_size, desc=f"Epoch {epoch}/{epochs}", leave=False):
            logits, cache = model.forward(xb)
            ce_loss, dlogits = cross_entropy_loss(logits, yb)

            grads = model.backward(dlogits, cache)
            model.add_l2_gradients(grads, weight_decay)

            l2 = 0.5 * weight_decay * model.l2_penalty()
            loss = ce_loss + l2

            model.apply_gradients(grads, lr=current_lr)

            preds = np.argmax(logits, axis=1)
            running_loss += float(loss) * len(xb)
            running_correct += int(np.sum(preds == yb))
            running_total += len(xb)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_metrics = evaluate_model(model, x_val, y_val)
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])

        history["epoch"].append(epoch)
        history["lr"].append(float(current_lr))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            meta = checkpoint_meta or {}
            save_checkpoint(model, checkpoint_path, meta)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stop_patience is not None and early_stop_patience > 0:
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_acc improvement for {early_stop_patience} epochs)."
                )
                stopped_early = True
                break

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "stopped_early": stopped_early,
        "last_epoch": history["epoch"][-1] if history["epoch"] else 0,
    }
