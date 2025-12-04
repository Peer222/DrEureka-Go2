from dataclasses import dataclass
from pathlib import Path
import tyro
import pickle
import pandas as pd

@dataclass
class Args:
    run_dir: Path
    "Directory containing metrics.pkl"


def print_metrics(pkl_file: Path):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        print(data)


if __name__ == "__main__":
    args = tyro.cli(Args)

    print_metrics(args.run_dir / "metrics.pkl")
    print_metrics(args.run_dir / "parameters.pkl")
