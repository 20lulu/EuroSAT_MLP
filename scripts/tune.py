# 搜索入口脚本：执行超参数搜索并汇总最优配置结果。

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_eurosat_splits
from src.search import run_hyperparameter_search
from src.utils import ensure_dir, now_str, print_table, save_json, set_seed


def parse_list(values: str, cast_fn):
    return [cast_fn(v.strip()) for v in values.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for NumPy MLP on EuroSAT")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)

    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--num-trials", type=int, default=20)

    parser.add_argument("--hidden1-dims", type=str, default="256,512")
    parser.add_argument("--hidden2-dims", type=str, default="128,256,512")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--learning-rates", type=str, default="0.01,0.0075,0.005")
    parser.add_argument("--lr-schedule", type=str, default="step", choices=["step", "exp"])
    parser.add_argument("--lr-step-sizes", type=str, default="15")
    parser.add_argument("--lr-gammas", type=str, default="0.5")
    parser.add_argument("--lr-decays", type=str, default="0.99,0.985,0.98")
    parser.add_argument("--weight-decays", type=str, default="1e-5,1e-4,5e-4")
    parser.add_argument("--grad-clips", type=str, default="0.0")
    parser.add_argument("--momentums", type=str, default="0.9")

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5)
    parser.add_argument("--augment-vflip-prob", type=float, default=0.0)
    parser.add_argument("--augment-rot90-prob", type=float, default=0.5)
    parser.add_argument("--augment-brightness-std", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = ensure_dir(Path(args.output_dir) / f"search_{now_str()}")

    print("Loading dataset once for all trials...")
    dataset = load_eurosat_splits(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        image_size=args.image_size,
        max_per_class=args.max_per_class,
    )

    base_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "image_size": args.image_size,
        "early_stop_patience": args.early_stop_patience,
        "lr_schedule": args.lr_schedule,
        "augment": args.augment,
        "augment_hflip_prob": args.augment_hflip_prob,
        "augment_vflip_prob": args.augment_vflip_prob,
        "augment_rot90_prob": args.augment_rot90_prob,
        "augment_brightness_std": args.augment_brightness_std,
    }

    search_space = {
        "hidden1_dim": parse_list(args.hidden1_dims, int),
        "hidden2_dim": parse_list(args.hidden2_dims, int),
        "activation": parse_list(args.activations, str),
        "learning_rate": parse_list(args.learning_rates, float),
        "lr_schedule": [args.lr_schedule],
        "lr_step_size": parse_list(args.lr_step_sizes, int),
        "lr_gamma": parse_list(args.lr_gammas, float),
        "weight_decay": parse_list(args.weight_decays, float),
        "grad_clip": parse_list(args.grad_clips, float),
        "augment": [args.augment],
        "augment_hflip_prob": [args.augment_hflip_prob],
        "augment_vflip_prob": [args.augment_vflip_prob],
        "augment_rot90_prob": [args.augment_rot90_prob],
        "augment_brightness_std": [args.augment_brightness_std],
        "momentum": parse_list(args.momentums, float),
        "batch_size": [args.batch_size],
        "epochs": [args.epochs],
    }
    if args.lr_schedule == "exp":
        search_space["lr_decay"] = parse_list(args.lr_decays, float)

    summary = run_hyperparameter_search(
        dataset=dataset,
        base_config=base_config,
        search_space=search_space,
        output_dir=run_dir,
        mode=args.mode,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    top5 = summary["results"][:5]
    print("Top search results (by best validation accuracy):")
    print_table(
        top5,
        keys=[
            "trial",
            "activation",
            "hidden1_dim",
            "hidden2_dim",
            "learning_rate",
            "lr_schedule",
            "lr_step_size",
            "lr_gamma",
            "lr_decay",
            "weight_decay",
            "momentum",
            "best_val_acc",
            "test_acc",
        ],
    )

    save_json(summary, Path(run_dir) / "search_results.json")
    print(f"Search output saved to: {run_dir}")
    print("Best trial:")
    print(summary["best"])


if __name__ == "__main__":
    main()
