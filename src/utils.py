# 通用工具模块：提供随机种子、路径管理、日志与文件读写等辅助函数。

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Set python/numpy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dictionary to json file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load dictionary from json file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays: Any) -> None:
    """Save numpy arrays to npz file."""
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(path, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load numpy arrays from npz file and return a plain dict."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def now_str() -> str:
    """Return compact local timestamp string."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_table(rows: list[dict[str, Any]], keys: list[str]) -> None:
    """Pretty-print a tiny list of dict rows in terminal."""
    if not rows:
        print("(empty)")
        return

    widths = {k: len(k) for k in keys}
    for row in rows:
        for k in keys:
            widths[k] = max(widths[k], len(str(row.get(k, ""))))

    def _fmt(row: dict[str, Any]) -> str:
        return " | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys)

    print(_fmt({k: k for k in keys}))
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        print(_fmt(row))


def to_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config values to JSON-serializable values."""
    output: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, Path):
            output[k] = str(v)
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            output[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            output[k] = int(v)
        elif isinstance(v, tuple):
            output[k] = list(v)
        else:
            output[k] = v
    return output


def abs_path(path: str | Path) -> str:
    """Return absolute path string."""
    return os.path.abspath(str(path))
