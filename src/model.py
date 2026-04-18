# 模型定义模块：实现三层 MLP 的参数初始化与前向/反向传播接口。

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim: int
    num_classes: int
    activation: str = "relu"
    seed: int = 42


class MLP:
    """A 3-layer MLP (input -> hidden -> output) with manual backprop."""

    def __init__(self, config: MLPConfig):
        self.config = config
        self.activation_name = config.activation.lower()
        if self.activation_name not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh")

        rng = np.random.default_rng(config.seed)
        h = config.hidden_dim
        d = config.input_dim
        c = config.num_classes

        # He for ReLU, Xavier for sigmoid/tanh.
        if self.activation_name == "relu":
            w1_scale = np.sqrt(2.0 / d)
        else:
            w1_scale = np.sqrt(1.0 / d)
        w2_scale = np.sqrt(1.0 / h)

        self.params: Dict[str, Array] = {
            "W1": rng.normal(0.0, w1_scale, size=(d, h)).astype(np.float32),
            "b1": np.zeros(h, dtype=np.float32),
            "W2": rng.normal(0.0, w2_scale, size=(h, c)).astype(np.float32),
            "b2": np.zeros(c, dtype=np.float32),
        }

    def _activate(self, x: Array) -> Array:
        if self.activation_name == "relu":
            return np.maximum(x, 0.0)
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        return np.tanh(x)

    def _activate_grad(self, z: Array, a: Array) -> Array:
        if self.activation_name == "relu":
            return (z > 0.0).astype(np.float32)
        if self.activation_name == "sigmoid":
            return a * (1.0 - a)
        return 1.0 - a * a

    def forward(self, x: Array) -> Tuple[Array, Dict[str, Array]]:
        """Forward pass, returns logits and cache used by backward."""
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]

        z1 = x @ w1 + b1
        a1 = self._activate(z1)
        logits = a1 @ w2 + b2
        cache = {"x": x, "z1": z1, "a1": a1}
        return logits, cache

    def backward(self, dlogits: Array, cache: Dict[str, Array]) -> Dict[str, Array]:
        """Backward pass from dLoss/dLogits to all parameters."""
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]

        w2 = self.params["W2"]

        grads: Dict[str, Array] = {}
        grads["W2"] = a1.T @ dlogits
        grads["b2"] = np.sum(dlogits, axis=0)

        da1 = dlogits @ w2.T
        dz1 = da1 * self._activate_grad(z1, a1)
        grads["W1"] = x.T @ dz1
        grads["b1"] = np.sum(dz1, axis=0)
        return grads

    def predict(self, x: Array, batch_size: int = 1024) -> Array:
        """Predict class ids for input array."""
        preds = []
        n = x.shape[0]
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            logits, _ = self.forward(x[s:e])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds, axis=0)

    def apply_gradients(self, grads: Dict[str, Array], lr: float) -> None:
        """SGD parameter update."""
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def l2_penalty(self) -> float:
        """Return sum of squared weights (bias terms excluded)."""
        return float(np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))

    def add_l2_gradients(self, grads: Dict[str, Array], weight_decay: float) -> None:
        """In-place add of L2 regularization gradients to weight params."""
        if weight_decay <= 0:
            return
        grads["W1"] += weight_decay * self.params["W1"]
        grads["W2"] += weight_decay * self.params["W2"]

    def state_dict(self) -> Dict[str, Array]:
        """Export parameters as a state dict."""
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, Array]) -> None:
        """Load parameters from state dict."""
        for k in self.params:
            if k not in state:
                raise KeyError(f"Missing key in state_dict: {k}")
            self.params[k] = state[k].astype(np.float32, copy=True)
