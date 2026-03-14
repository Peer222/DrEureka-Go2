import tyro
from dataclasses import dataclass
from typing import List, Literal
import re
import pandas as pd
from pathlib import Path
import plots_plus


def construct_metrics_df(run_log_path: Path) -> pd.DataFrame:
    with open(run_log_path, "r") as f:
        logged_lines = f.readlines()
    run_log = {}
    for i, line in enumerate(logged_lines):
        if line.startswith("│") and line.endswith("│\n"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "train/episode/rew success/mean":
                key = "fitness_score"
            elif key == "timesteps" or key == "iterations":
                key = key
            elif "train/episode/rew" in key:
                key = key.split("/")[2].split("rew ")[-1]
            elif key == "train/episode/episode length/mean":
                key = "episode_length"
            elif "loss" in key:
                key = key.split("/")[0].split("mean ")[-1]

            run_log[key] = run_log.get(key, []) + [float(val)]

    logged_data_df = pd.DataFrame(run_log)
    print(logged_data_df)
    if "fitness_score" not in run_log.keys():
        raise Exception("'fitness_score' is missing in run log!")

    return logged_data_df


def create_plots(
    run_log_path: Path,
    graphics_dir: Path,
):
    graphics_dir.mkdir(exist_ok=True, parents=True)

    metrics_df = construct_metrics_df(run_log_path)
    plots_plus.lineplot(
        metrics_df,
        x="iterations",
        y="episode_length",
        ylim=(0, None),
        filepath=graphics_dir / "episode_length.png",
    )

    losses = metrics_df.columns[metrics_df.columns.str.match(".*loss")]
    losses = {key: key.split(" loss")[0] for key in losses}
    losses_df: pd.DataFrame = metrics_df[list(losses.keys()) + ["iterations"]].rename(losses, axis=1)  # type: ignore
    losses_df = plots_plus.utils.rotate_df(
        losses_df, "iterations", losses.values(), "loss"
    )
    plots_plus.lineplot(
        losses_df,
        x="iterations",
        y="loss",
        hue="type",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "losses.png",
    )

    rewards = metrics_df.columns[metrics_df.columns.str.match(".*reward")]
    rewards = {key: key.split(" reward")[0] for key in rewards}
    rewards_df: pd.DataFrame = metrics_df[list(rewards.keys()) + ["iterations"]].rename(rewards, axis=1)  # type: ignore
    rewards_df = plots_plus.utils.rotate_df(
        rewards_df, "iterations", rewards.values(), "reward"
    )
    plots_plus.lineplot(
        rewards_df,
        x="iterations",
        y="reward",
        hue="type",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "rewards.png",
    )

    plots_plus.lineplot(
        metrics_df,
        x="iterations",
        y="fitness_score",
        ylim=(-10, None),
        colorpalette=plots_plus.colors.ITERATION_COLOR_MAP,
        filepath=graphics_dir / "fitness_score.png",
    )


__all__ = ["create_plots"]


def __dir__():
    return __all__


if __name__ == "__main__":

    @dataclass
    class Args:
        result_dir: Path
        """directory in which graphics are stored"""
        run_log_path: Path
        """Path to run directory"""

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    create_plots(args.run_log_path, args.result_dir)
