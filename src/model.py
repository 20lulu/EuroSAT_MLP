# 模型定义模块：实现三层 MLP 的参数初始化与前向/反向传播接口。

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden1_dim: int = 512
    hidden2_dim: int | None = 256
    activation: str = "relu"
    seed: int = 42


class MLP:
    """An MLP with one or two hidden layers and manual backprop."""

    def __init__(self, config: MLPConfig):
        self.config = config
        self.activation_name = config.activation.lower()
        if self.activation_name not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh")

        rng = np.random.default_rng(config.seed)
        d = config.input_dim
        c = config.num_classes
        h1 = int(config.hidden1_dim)
        h2 = None if config.hidden2_dim is None else int(config.hidden2_dim)
        if h1 <= 0:
            raise ValueError("hidden1_dim must be > 0")
        if h2 is not None and h2 <= 0:
            raise ValueError("hidden2_dim must be > 0 when provided")
        self.has_second_hidden = h2 is not None

        # He for ReLU, Xavier for sigmoid/tanh.
        if self.activation_name == "relu":
            w1_scale = np.sqrt(2.0 / d)
            hidden_gain = 2.0
        else:
            w1_scale = np.sqrt(1.0 / d)
            hidden_gain = 1.0

        params: Dict[str, Array] = {
            "W1": rng.normal(0.0, w1_scale, size=(d, h1)).astype(np.float32),
            "b1": np.zeros(h1, dtype=np.float32),
        }
        if self.has_second_hidden:
            assert h2 is not None
            w2_scale = np.sqrt(hidden_gain / h1)
            w3_scale = np.sqrt(1.0 / h2)
            params.update(
                {
                    "W2": rng.normal(0.0, w2_scale, size=(h1, h2)).astype(np.float32),
                    "b2": np.zeros(h2, dtype=np.float32),
                    "W3": rng.normal(0.0, w3_scale, size=(h2, c)).astype(np.float32),
                    "b3": np.zeros(c, dtype=np.float32),
                }
            )
        else:
            w2_scale = np.sqrt(1.0 / h1)
            params.update(
                {
                    "W2": rng.normal(0.0, w2_scale, size=(h1, c)).astype(np.float32),
                    "b2": np.zeros(c, dtype=np.float32),
                }
            )
        self.params = params

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
        z1 = x @ w1 + b1
        a1 = self._activate(z1)
        if self.has_second_hidden:
            w2, b2 = self.params["W2"], self.params["b2"]
            w3, b3 = self.params["W3"], self.params["b3"]
            z2 = a1 @ w2 + b2
            a2 = self._activate(z2)
            logits = a2 @ w3 + b3
            cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        else:
            w2, b2 = self.params["W2"], self.params["b2"]
            logits = a1 @ w2 + b2
            cache = {"x": x, "z1": z1, "a1": a1}
        return logits, cache

    def backward(self, dlogits: Array, cache: Dict[str, Array]) -> Dict[str, Array]:
        """Backward pass from dLoss/dLogits to all parameters."""
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]

        grads: Dict[str, Array] = {}
        if self.has_second_hidden:
            z2 = cache["z2"]
            a2 = cache["a2"]
            w2 = self.params["W2"]
            w3 = self.params["W3"]

            grads["W3"] = a2.T @ dlogits
            grads["b3"] = np.sum(dlogits, axis=0)

            da2 = dlogits @ w3.T
            dz2 = da2 * self._activate_grad(z2, a2)
            grads["W2"] = a1.T @ dz2
            grads["b2"] = np.sum(dz2, axis=0)

            da1 = dz2 @ w2.T
            dz1 = da1 * self._activate_grad(z1, a1)
            grads["W1"] = x.T @ dz1
            grads["b1"] = np.sum(dz1, axis=0)
        else:
            w2 = self.params["W2"]
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
        penalty = float(np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        if "W3" in self.params:
            penalty += float(np.sum(self.params["W3"] ** 2))
        return penalty

    def add_l2_gradients(self, grads: Dict[str, Array], weight_decay: float) -> None:
        """In-place add of L2 regularization gradients to weight params."""
        if weight_decay <= 0:
            return
        for name in ("W1", "W2", "W3"):
            if name in grads:
                grads[name] += weight_decay * self.params[name]

    def state_dict(self) -> Dict[str, Array]:
        """Export parameters as a state dict."""
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, Array]) -> None:
        """Load parameters from state dict."""
        for k in self.params:
            if k not in state:
                raise KeyError(f"Missing key in state_dict: {k}")
            self.params[k] = state[k].astype(np.float32, copy=True)
