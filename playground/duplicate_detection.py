from typing import Union, Optional, Tuple, List, Literal, Dict
from pathlib import Path
import pandas as pd
import ast
import re
import sys
from dataclasses import dataclass
import logging
import python_minifier
import plots_plus


@dataclass
class Args:
    runs: List[Path]
    """Path to stats.csv files from eureka runs (multiple seeds)"""
    resultdir: Optional[Path] = Path("duplicate_detection")
    """Result directory. Has to be specified if multiple runs are given"""


def get_rewards(reward_dir: Path) -> pd.DataFrame:
    reward_files = reward_dir.glob("*.py")
    rewards = []
    for reward_file in reward_files:
        match = re.match(r"[^\d]*(\d+)[^\d]*(\d+)", reward_file.stem)
        if match is None:
            raise Exception(f"Unknown file pattern found: {reward_file.stem}")
        match.group(1)  # iteration

        # read lines and remove comments and strip the code
        stripped = ""
        reward_code = ""
        with open(reward_file, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                reward_code += line
                stripped_line = line.split("#")[0]
                stripped += stripped_line

        logging.debug(stripped)

        rewards.append(
            {
                "iteration": int(match.group(1)),
                "sample": int(match.group(2)),
                "original": reward_code,
                "stripped": stripped,
            }
        )
    rewards_df: pd.DataFrame = pd.DataFrame(rewards).sort_values(
        ["iteration", "sample"]
    )
    rewards_df.reset_index(inplace=True)
    return rewards_df


def detect_duplicates(
    args: Args, stats_df: pd.DataFrame, rewards_df: pd.DataFrame
):
    keys = ["task", "version", "seed", "iteration", "sample"]
    duplicates = (
        stats_df
        .groupby(keys)
        .size()
        .reset_index(name="count")
        .query("count > 1")
    )

    print(duplicates)
    merged_df = pd.merge(stats_df, rewards_df, on=keys)
    experiment_groups = merged_df.groupby(["task", "version", "seed"])


    for group, experiment in experiment_groups:
        iteration_groups = experiment.groupby("iteration")
        best_reward = ""
        duplicates = {}
        for iteration, samples in iteration_groups:
            duplicates[iteration] = 0
            max_idx = samples["fitness_score_max"].idxmax()
            if iteration == 0:
                best_reward = samples.loc[max_idx, "original"]
                continue

            for _, sample in samples.iterrows():
                if sample["original"] == best_reward:
                    duplicates[iteration] += 1
            
            best_reward = samples.loc[max_idx, "original"]
        
        logging.info(f"{group}: {duplicates}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import tyro

    args = tyro.cli(Args, description="Complexity Analysis based on reward code length")
    assert (
        len(args.runs) > 1 and args.resultdir or len(args.runs) == 1
    ), "If multiple runs are shown, a result dir must be specified"

    stats_df = pd.DataFrame()
    rewards_df = pd.DataFrame()
    for run_dir in args.runs:
        _stats_df = pd.read_csv(run_dir / "stats.csv")

        version_order = []
        match = re.search(".*/([^_]+)_.*", _stats_df["version"].iloc[0])
        if match is None:
            raise Exception(
                f"Unknown model version found: {_stats_df['version'].iloc[0]}"
            )
        _stats_df["version"] = match.group(1)
        if "GW" in run_dir.stem:
            _stats_df["task"] = "Ball Balancing"
        elif "FL" in run_dir.stem:
            _stats_df["task"] = "Forward Locomotion"
        else:
            raise NotImplementedError(
                f"{run_dir.stem} needs to include either FL or GW"
            )
        if "sample" not in _stats_df.columns:
            _stats_df["sample"] = [i for i in range(16) for _ in range(5)]
        stats_df = pd.concat([stats_df, _stats_df])

        ### rewards
        new_rewards_df = get_rewards(run_dir / "rewards")
        rewards_df = pd.concat([rewards_df, new_rewards_df])

    stats_df.reset_index(inplace=True)

    rewards_df.reset_index(inplace=True)
    rewards_df["seed"] = stats_df["seed"]
    rewards_df["version"] = stats_df["version"]
    rewards_df["task"] = stats_df["task"]

    resultdir = args.resultdir if args.resultdir else args.runs[0]
    resultdir.mkdir(exist_ok=True)

    detect_duplicates(args, stats_df, rewards_df)

    if args.resultdir:
        with open(args.resultdir / "command.txt", "w") as f:
            f.write(" ".join([sys.executable, *sys.argv]))
