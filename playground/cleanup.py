from pathlib import Path
import shutil
from dataclasses import dataclass
import tyro
import pandas as pd


@dataclass
class Args:
    eureka_run_dir: Path
    """Path to eureka run base directory"""
    ckpt_fitness_threshold: int
    """Threshold at which checkpoints are deleted (lower)"""


def delete_logs(base_dir: Path):
    log_files = base_dir.glob("**/outputs.log")
    for log_file in log_files:
        log_file.unlink()


def delete_checkpoints(base_dir: Path, fitness_threshold: int):
    stats = pd.read_csv(base_dir / "stats.csv")
    successful_executions = stats[stats["execution"] == 1].reset_index()
    underperforming_runs: pd.DataFrame = successful_executions[successful_executions["fitness_score_max"] < fitness_threshold]  # type: ignore

    checkpoint_dirs = list(base_dir.glob("**/checkpoints/"))
    checkpoint_dirs.sort()

    for i, chkpt_dir in enumerate(checkpoint_dirs):
        if i in underperforming_runs.index:
            shutil.rmtree(chkpt_dir)


def delete_wandb_runs(base_dir: Path):
    wandb_dirs = base_dir.glob("./*/*/wandb/")
    for wandb_dir in wandb_dirs:
        shutil.rmtree(wandb_dir)


def delete_submitit(base_dir: Path):
    submitit_dir = base_dir / "submitit"
    if submitit_dir.exists():
        shutil.rmtree(submitit_dir)


if __name__ == "__main__":
    args = tyro.cli(Args)
    delete_logs(args.eureka_run_dir)
    delete_wandb_runs(args.eureka_run_dir)
    delete_checkpoints(args.eureka_run_dir, args.ckpt_fitness_threshold)
    delete_submitit(args.eureka_run_dir)
