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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)

    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--num-trials", type=int, default=20)

    parser.add_argument("--hidden-dims", type=str, default="128,256,512")
    parser.add_argument("--activations", type=str, default="relu,sigmoid,tanh")
    parser.add_argument("--learning-rates", type=str, default="0.1,0.05,0.03,0.01")
    parser.add_argument("--lr-decays", type=str, default="0.99,0.985,0.98")
    parser.add_argument("--weight-decays", type=str, default="0.0,1e-4,5e-4,1e-3")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=10)
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
    }

    search_space = {
        "hidden_dim": parse_list(args.hidden_dims, int),
        "activation": parse_list(args.activations, str),
        "learning_rate": parse_list(args.learning_rates, float),
        "lr_decay": parse_list(args.lr_decays, float),
        "weight_decay": parse_list(args.weight_decays, float),
        "batch_size": [args.batch_size],
        "epochs": [args.epochs],
    }

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
            "hidden_dim",
            "learning_rate",
            "lr_decay",
            "weight_decay",
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
