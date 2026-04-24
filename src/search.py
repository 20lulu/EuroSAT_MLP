# 超参数搜索模块：管理不同超参数组合的训练与结果对比。

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np

from src.evaluate import evaluate_model
from src.model import MLP, MLPConfig
from src.trainer import build_model_from_checkpoint, train_model
from src.utils import ensure_dir, save_json


SearchMode = Literal["grid", "random"]


def _expand_grid(search_space: Dict[str, List[object]]) -> List[Dict[str, object]]:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combos = []
    for product_values in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, product_values)})
    return combos


def run_hyperparameter_search(
    dataset: Dict[str, object],
    base_config: Dict[str, object],
    search_space: Dict[str, List[object]],
    output_dir: str | Path,
    mode: SearchMode = "grid",
    num_trials: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    """Run grid/random search and return best trial summary."""
    output_dir = ensure_dir(output_dir)

    all_candidates = _expand_grid(search_space)
    if not all_candidates:
        raise ValueError("Search space cannot be empty")

    if mode == "random":
        rng = np.random.default_rng(seed)
        n = min(num_trials, len(all_candidates))
        sel = rng.choice(len(all_candidates), size=n, replace=False)
        candidates = [all_candidates[i] for i in sel]
    else:
        candidates = all_candidates

    x_train = dataset["train"]["X"]
    y_train = dataset["train"]["y"]
    x_val = dataset["val"]["X"]
    y_val = dataset["val"]["y"]
    x_test = dataset["test"]["X"]
    y_test = dataset["test"]["y"]
    class_names = dataset["class_names"]

    results: List[Dict[str, object]] = []
    best = None

    for trial_id, hp in enumerate(candidates, start=1):
        trial_cfg = {**base_config, **hp}
        trial_dir = ensure_dir(Path(output_dir) / f"trial_{trial_id:03d}")
        ckpt_path = trial_dir / "best_model.npz"
        hidden1_dim = int(trial_cfg.get("hidden1_dim", trial_cfg.get("hidden_dim", 256)))
        raw_hidden2 = trial_cfg.get("hidden2_dim", None)
        hidden2_dim = None if raw_hidden2 is None or int(raw_hidden2) <= 0 else int(raw_hidden2)

        model_cfg = MLPConfig(
            input_dim=x_train.shape[1],
            num_classes=len(class_names),
            hidden1_dim=hidden1_dim,
            hidden2_dim=hidden2_dim,
            activation=str(trial_cfg["activation"]),
            seed=int(trial_cfg.get("seed", seed)) + trial_id,
        )
        model = MLP(model_cfg)

        meta = {
            "model": {
                "input_dim": int(model_cfg.input_dim),
                "hidden1_dim": int(model_cfg.hidden1_dim),
                "hidden2_dim": None if model_cfg.hidden2_dim is None else int(model_cfg.hidden2_dim),
                "num_classes": int(model_cfg.num_classes),
                "activation": model_cfg.activation,
            },
            "class_names": class_names,
            "input_shape": list(dataset["input_shape"]),
            "norm_mean": dataset["norm_mean"].tolist(),
            "norm_std": dataset["norm_std"].tolist(),
            "split": {
                "val_ratio": float(base_config["val_ratio"]),
                "test_ratio": float(base_config["test_ratio"]),
                "seed": int(base_config["seed"]),
                "image_size": int(base_config["image_size"]),
            },
            "hparams": trial_cfg,
        }

        out = train_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=int(trial_cfg["epochs"]),
            batch_size=int(trial_cfg["batch_size"]),
            learning_rate=float(trial_cfg["learning_rate"]),
            lr_schedule=str(trial_cfg.get("lr_schedule", "step")),
            lr_step_size=int(trial_cfg.get("lr_step_size", 15)),
            lr_gamma=float(trial_cfg.get("lr_gamma", 0.5)),
            lr_decay=float(trial_cfg.get("lr_decay", 0.98)),
            weight_decay=float(trial_cfg["weight_decay"]),
            checkpoint_path=ckpt_path,
            checkpoint_meta=meta,
            seed=int(trial_cfg.get("seed", seed)),
            early_stop_patience=int(trial_cfg.get("early_stop_patience", 10)),
            grad_clip=float(trial_cfg.get("grad_clip", 0.0)),
            momentum=float(trial_cfg.get("momentum", 0.9)),
            augment=bool(trial_cfg.get("augment", True)),
            input_shape=tuple(dataset["input_shape"]),
            augment_hflip_prob=float(trial_cfg.get("augment_hflip_prob", 0.5)),
            augment_vflip_prob=float(trial_cfg.get("augment_vflip_prob", 0.0)),
            augment_rot90_prob=float(trial_cfg.get("augment_rot90_prob", 0.5)),
            augment_brightness_std=float(trial_cfg.get("augment_brightness_std", 0.05)),
        )

        best_model, _ = build_model_from_checkpoint(ckpt_path)
        val_metrics = evaluate_model(best_model, x_val, y_val)
        test_metrics = evaluate_model(best_model, x_test, y_test)

        row = {
            "trial": trial_id,
            "hidden1_dim": hidden1_dim,
            "hidden2_dim": hidden2_dim,
            "activation": str(trial_cfg["activation"]),
            "learning_rate": float(trial_cfg["learning_rate"]),
            "lr_schedule": str(trial_cfg.get("lr_schedule", "step")),
            "lr_step_size": int(trial_cfg.get("lr_step_size", 15)),
            "lr_gamma": float(trial_cfg.get("lr_gamma", 0.5)),
            "lr_decay": float(trial_cfg.get("lr_decay", 0.98)),
            "weight_decay": float(trial_cfg["weight_decay"]),
            "batch_size": int(trial_cfg["batch_size"]),
            "epochs": int(trial_cfg["epochs"]),
            "grad_clip": float(trial_cfg.get("grad_clip", 0.0)),
            "momentum": float(trial_cfg.get("momentum", 0.9)),
            "augment": bool(trial_cfg.get("augment", True)),
            "early_stop_patience": int(trial_cfg.get("early_stop_patience", 10)),
            "best_epoch": int(out["best_epoch"]),
            "best_val_acc": float(out["best_val_acc"]),
            "val_acc": float(val_metrics["accuracy"]),
            "test_acc": float(test_metrics["accuracy"]),
            "stopped_early": bool(out["stopped_early"]),
            "last_epoch": int(out["last_epoch"]),
            "checkpoint": str(ckpt_path),
        }
        results.append(row)

        save_json(row, trial_dir / "summary.json")

        if best is None or row["best_val_acc"] > best["best_val_acc"]:
            best = row

        print(
            f"[Trial {trial_id:03d}/{len(candidates)}] "
            f"val_acc={row['val_acc']:.4f}, test_acc={row['test_acc']:.4f}, "
            f"best_val={row['best_val_acc']:.4f}"
        )

    if best is None:
        raise RuntimeError("Hyperparameter search produced no results")

    results_sorted = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    summary = {
        "mode": mode,
        "num_candidates": len(candidates),
        "best": best,
        "results": results_sorted,
    }
    save_json(summary, Path(output_dir) / "search_results.json")
    return summary
