from typing import Union, Optional, Tuple, List, Literal, Dict
from pathlib import Path
import pandas as pd
import ast
import re
import sys
from dataclasses import dataclass
import logging
import plots_plus


@dataclass
class Args:
    runs: List[Path]
    """Path to stats.csv files from eureka runs (multiple seeds)"""
    resultdir: Optional[Path] = None
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
        content = ""
        with open(reward_file, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                stripped_line = line.strip().split("#")[0]
                content += stripped_line + "\n"

        logging.debug(content)
        rewards.append(
            {
                "iteration": int(match.group(1)),
                "sample": int(match.group(2)),
                "content": content,
            }
        )
    rewards_df: pd.DataFrame = pd.DataFrame(rewards).sort_values(
        ["iteration", "sample"]
    )
    rewards_df.reset_index(inplace=True)
    return rewards_df


def analyze_reward_complexity(args: Args, stats_df: pd.DataFrame, rewards_df: pd.DataFrame):
    complexity_data = []
    for index, sample in stats_df.iterrows():
        reward = rewards_df.iloc[index]["content"]

        complexity_data.append({
            "task": sample["task"],
            "version": sample["version"],
            "seed": sample["seed"],
            "iteration": sample["iteration"],
            "sample": sample["sample"],
            "fitness_score_max": sample["fitness_score_max"],
            "reward_length": len(reward),
        })

    complexity_df = pd.DataFrame(complexity_data)
    version_order = complexity_df["version"].drop_duplicates().to_list()
    complexity_df.to_csv(args.resultdir / "reward_complexity.csv")  # type: ignore

    logging.info(complexity_df["reward_length"].describe())
    plots_plus.gridlineplot(
        complexity_df,
        x="iteration",
        y="reward_length",
        hue="version",
        axes="task",
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        hue_order=version_order,
        ylim=(0, None),
        alpha=0.75,
        filepath=args.resultdir / "complexity.png",  # type: ignore
    )
    bb_complexity_df = complexity_df[complexity_df["task"] == "Ball Balancing"]
    correlation = bb_complexity_df[["fitness_score_max", "reward_length"]].corr("spearman")  # type: ignore
    logging.info(f"Ball Balancing: Reward length - fitness score correlation (spearman): {correlation}")
    plots_plus.scatterplot(bb_complexity_df, "reward_length", "fitness_score_max", hue="version", filepath=args.resultdir / "bb_fitness_complexity.png")  # type: ignore

    fl_complexity_df = complexity_df[complexity_df["task"] == "Forward Locomotion"]
    correlation = fl_complexity_df[["fitness_score_max", "reward_length"]].corr("spearman")  # type: ignore
    logging.info(f"Forward Locomotion: Reward length - fitness score correlation (spearman): {correlation}")
    plots_plus.scatterplot(fl_complexity_df, "reward_length", "fitness_score_max", hue="version", filepath=args.resultdir / "fl_fitness_complexity.png")  # type: ignore
    return complexity_df

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import tyro

    args = tyro.cli(Args, description="Complexity Analysis based on reward code length")
    assert len(args.runs) > 1 and args.resultdir or len(args.runs) == 1, "If multiple runs are shown, a result dir must be specified"

    stats_df = pd.DataFrame()
    rewards_df = pd.DataFrame()
    for run_dir in args.runs:
        _stats_df = pd.read_csv(run_dir / "stats.csv")

        version_order = []
        match = re.search(".*/([^_]+)_.*", _stats_df["version"].iloc[0])
        if match is None:
            raise Exception(f"Unknown model version found: {_stats_df['version'].iloc[0]}")
        version_order.append(match.group(1))
        _stats_df["version"] = match.group(1)
        if "GW" in run_dir.stem:
            _stats_df["task"] = "Ball Balancing"
        elif "FL" in run_dir.stem:
            _stats_df["task"] = "Forward Locomotion"
        else:
            raise NotImplementedError(
                f"{run_dir.stem} needs to include either FL or GW"
            )
        stats_df = pd.concat([stats_df, _stats_df])

        ### rewards
        new_rewards_df = get_rewards(run_dir / "rewards")
        rewards_df = pd.concat([rewards_df, new_rewards_df])

    stats_df.reset_index(inplace=True)

    rewards_df.reset_index(inplace=True)
    rewards_df["seed"] = stats_df["seed"]
    rewards_df["version"] = stats_df["version"]

    resultdir = args.resultdir if args.resultdir else args.runs[0]
    resultdir.mkdir(exist_ok=True)

    analyze_reward_complexity(args, stats_df, rewards_df)

    if args.resultdir:
        with open(args.resultdir / "command.txt", "w") as f:
            f.write(" ".join([sys.executable, *sys.argv]))
