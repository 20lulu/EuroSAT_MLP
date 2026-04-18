# 数据加载与预处理模块：读取 EuroSAT 数据、划分数据集并完成标准化等处理。

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


Array = np.ndarray


def _collect_samples(
    data_dir: str | Path,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found under: {data_dir}")

    rng = np.random.default_rng(seed)
    paths: list[str] = []
    labels: list[int] = []

    for idx, class_name in enumerate(class_names):
        folder = data_dir / class_name
        images = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if max_per_class is not None and len(images) > max_per_class:
            sel = rng.choice(len(images), size=max_per_class, replace=False)
            images = [images[i] for i in sorted(sel)]

        for p in images:
            paths.append(str(p))
            labels.append(idx)

    return np.array(paths), np.array(labels, dtype=np.int64), class_names


def _load_image_batch(paths: np.ndarray, image_size: int, desc: str) -> Array:
    n = len(paths)
    h = w = image_size
    out = np.empty((n, h, w, 3), dtype=np.float32)

    for i, p in enumerate(tqdm(paths, desc=desc, leave=False)):
        with Image.open(p) as img:
            img = img.convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def _normalize(
    x_train: Array,
    x_val: Array,
    x_test: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.clip(std, 1e-6, None)

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std

    return x_train_n, x_val_n, x_test_n, mean.reshape(-1), std.reshape(-1)


def load_eurosat_splits(
    data_dir: str | Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    image_size: int = 64,
    max_per_class: Optional[int] = None,
) -> Dict[str, object]:
    """Load EuroSAT, split into train/val/test, normalize, then flatten features."""
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Require 0 < val_ratio, test_ratio and val_ratio + test_ratio < 1")

    paths, labels, class_names = _collect_samples(data_dir, max_per_class=max_per_class, seed=seed)
    indices = np.arange(len(paths))
    num_classes = len(class_names)

    class_counts = np.bincount(labels, minlength=num_classes)
    if int(class_counts.min()) < 3:
        raise ValueError(
            "Each class must have at least 3 samples for stratified train/val/test split. "
            f"Current minimum class count: {int(class_counts.min())}"
        )

    n_total = len(indices)
    n_temp = int(np.ceil(n_total * (val_ratio + test_ratio)))
    n_test = int(np.ceil(n_temp * (test_ratio / (val_ratio + test_ratio))))
    n_val = n_temp - n_test
    if n_temp < num_classes or n_val < num_classes or n_test < num_classes:
        raise ValueError(
            "Split sizes are too small for stratified splitting. "
            "Try increasing dataset size (or max_per_class), or adjust val/test ratios."
        )

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    test_fraction_in_temp = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction_in_temp,
        random_state=seed,
        stratify=labels[temp_idx],
        shuffle=True,
    )

    train_paths, val_paths, test_paths = paths[train_idx], paths[val_idx], paths[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    x_train = _load_image_batch(train_paths, image_size, desc="Loading train")
    x_val = _load_image_batch(val_paths, image_size, desc="Loading val")
    x_test = _load_image_batch(test_paths, image_size, desc="Loading test")

    x_train, x_val, x_test, mean, std = _normalize(x_train, x_val, x_test)

    input_shape = x_train.shape[1:]
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_val = x_val.reshape(x_val.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    return {
        "class_names": class_names,
        "input_shape": tuple(int(v) for v in input_shape),
        "norm_mean": mean.astype(np.float32),
        "norm_std": std.astype(np.float32),
        "train": {"X": x_train, "y": y_train, "paths": train_paths},
        "val": {"X": x_val, "y": y_val, "paths": val_paths},
        "test": {"X": x_test, "y": y_test, "paths": test_paths},
    }


def iterate_minibatches(
    x: Array,
    y: Array,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
) -> Iterator[Tuple[Array, Array]]:
    """Yield mini-batches for SGD training."""
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(idx)

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        batch_idx = idx[s:e]
        yield x[batch_idx], y[batch_idx]
