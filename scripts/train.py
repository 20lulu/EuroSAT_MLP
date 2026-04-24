# 训练入口脚本：加载配置并启动单次模型训练流程。

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
from src.model import MLP, MLPConfig
from src.trainer import build_model_from_checkpoint, train_model
from src.utils import ensure_dir, now_str, save_json, set_seed, to_serializable_config
from src.visualize import plot_confusion_matrix, plot_first_layer_weights, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NumPy MLP on EuroSAT")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for dataset split. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Seed for model parameter initialization. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Seed for training-time randomness (shuffle/augmentation). Defaults to --seed when omitted.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)

    parser.add_argument("--hidden1-dim", type=int, default=512)
    parser.add_argument("--hidden2-dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--lr-schedule", type=str, default="step", choices=["step", "exp"])
    parser.add_argument("--lr-step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5)
    parser.add_argument("--augment-vflip-prob", type=float, default=0.0)
    parser.add_argument("--augment-rot90-prob", type=float, default=0.5)
    parser.add_argument("--augment-brightness-std", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    train_seed = args.seed if args.train_seed is None else args.train_seed
    set_seed(train_seed)

    run_dir = ensure_dir(Path(args.output_dir) / f"train_{now_str()}")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    fig_dir = ensure_dir(run_dir / "figures")

    print("Loading dataset...")
    dataset = load_eurosat_splits(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=split_seed,
        image_size=args.image_size,
        max_per_class=args.max_per_class,
    )

    x_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
    x_val, y_val = dataset["val"]["X"], dataset["val"]["y"]
    x_test, y_test = dataset["test"]["X"], dataset["test"]["y"]
    class_names = dataset["class_names"]

    model_cfg = MLPConfig(
        input_dim=x_train.shape[1],
        num_classes=len(class_names),
        hidden1_dim=args.hidden1_dim,
        hidden2_dim=(None if args.hidden2_dim <= 0 else args.hidden2_dim),
        activation=args.activation,
        seed=model_seed,
    )
    model = MLP(model_cfg)

    checkpoint_path = ckpt_dir / "best_model.npz"
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
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": split_seed,
            "image_size": args.image_size,
        },
        "seeds": {
            "base_seed": args.seed,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "train_seed": train_seed,
        },
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lr_schedule": args.lr_schedule,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "lr_decay": args.lr_decay,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "momentum": args.momentum,
            "early_stop_patience": args.early_stop_patience,
            "hidden1_dim": args.hidden1_dim,
            "hidden2_dim": None if args.hidden2_dim <= 0 else args.hidden2_dim,
            "activation": args.activation,
            "augment": args.augment,
            "augment_hflip_prob": args.augment_hflip_prob,
            "augment_vflip_prob": args.augment_vflip_prob,
            "augment_rot90_prob": args.augment_rot90_prob,
            "augment_brightness_std": args.augment_brightness_std,
        },
    }

    print("Training...")
    train_out = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        checkpoint_path=checkpoint_path,
        checkpoint_meta=meta,
        seed=train_seed,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        momentum=args.momentum,
        augment=args.augment,
        input_shape=tuple(dataset["input_shape"]),
        augment_hflip_prob=args.augment_hflip_prob,
        augment_vflip_prob=args.augment_vflip_prob,
        augment_rot90_prob=args.augment_rot90_prob,
        augment_brightness_std=args.augment_brightness_std,
    )

    best_model, _ = build_model_from_checkpoint(checkpoint_path)

    train_metrics = evaluate_model(best_model, x_train, y_train)
    val_metrics = evaluate_model(best_model, x_val, y_val)
    test_metrics = evaluate_model(best_model, x_test, y_test)

    y_pred_test = predict(best_model, x_test)
    details = confusion_and_report(y_test, y_pred_test, class_names)
    cm = details["confusion_matrix"]

    mis_records = collect_misclassified(
        y_true=y_test,
        y_pred=y_pred_test,
        paths=dataset["test"]["paths"],
        class_names=class_names,
        max_items=36,
    )

    plot_training_curves(train_out["history"], fig_dir / "training_curves.png")
    plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix_norm.png", normalize=True)
    plot_first_layer_weights(best_model.params["W1"], dataset["input_shape"], fig_dir / "first_layer_weights.png")
    plot_misclassified_grid(mis_records, fig_dir / "misclassified_examples.png", max_items=36)
    save_misclassified_json(mis_records, run_dir / "misclassified.json")

    summary = {
        "config": to_serializable_config(vars(args)),
        "best_epoch": train_out["best_epoch"],
        "best_val_acc": train_out["best_val_acc"],
        "stopped_early": train_out["stopped_early"],
        "last_epoch": train_out["last_epoch"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": details["classification_report"],
        "checkpoint_path": str(checkpoint_path),
    }

    save_json(train_out["history"], run_dir / "history.json")
    save_json(summary, run_dir / "summary.json")
    save_json(meta, run_dir / "run_meta.json")

    print("=" * 70)
    print(f"Run dir: {run_dir}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best epoch: {train_out['best_epoch']}")
    print(f"Val accuracy(best): {train_out['best_val_acc']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
