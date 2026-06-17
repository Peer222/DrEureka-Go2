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
        stripped = ""
        reward_code = ""
        compressed_code = ""
        with open(reward_file, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                reward_code += line
                stripped_line = line.strip().split("#")[0]
                stripped += stripped_line + "\n"
                if "import *" not in line :
                    compressed_code += line

        logging.debug(stripped)
        try:
            compressed = python_minifier.minify(
                compressed_code,
                remove_annotations=True,
                remove_literal_statements=True,
                rename_globals=True,
            )
            compressed = compressed.replace("torch.", "")
        except:
            compressed = ""
        logging.debug(compressed)

        rewards.append(
            {
                "iteration": int(match.group(1)),
                "sample": int(match.group(2)),
                "original": reward_code,
                "stripped": stripped,
                "compressed": compressed,
            }
        )
    rewards_df: pd.DataFrame = pd.DataFrame(rewards).sort_values(
        ["iteration", "sample"]
    )
    rewards_df.reset_index(inplace=True)
    return rewards_df


def analyze_reward_complexity(
    args: Args, stats_df: pd.DataFrame, rewards_df: pd.DataFrame
):
    complexity_data = []
    for index, sample in stats_df.iterrows():
        stripped_reward = rewards_df.iloc[index]["stripped"]
        compressed_reward = rewards_df.iloc[index]["compressed"]
        original_reward = rewards_df.iloc[index]["original"]
        # skips invalid code
        if not len(compressed_reward):
            continue

        complexity_data.append(
            {
                "task": sample["task"],
                "version": sample["version"],
                "seed": sample["seed"],
                "iteration": sample["iteration"],
                "sample": sample["sample"],
                "num_reward_functions": sample["num_reward_functions"],
                "fitness_score_max": sample["fitness_score_max"],
                "original_reward_length": len(original_reward),
                "stripped_reward_length": len(stripped_reward),
                "compressed_reward_length": len(compressed_reward),
                "compressed_reward_length_per_component": len(compressed_reward) / max(sample["num_reward_functions"], 1),
            }
        )

    complexity_df = pd.DataFrame(complexity_data)
    version_order = complexity_df["version"].drop_duplicates().to_list()
    complexity_df.to_csv(args.resultdir / "reward_complexity.csv")  # type: ignore

    for type in ["stripped_reward_length", "compressed_reward_length", "original_reward_length", "compressed_reward_length_per_component"]:
        prefix = type.split("_")[0] if type != "compressed_reward_length_per_component" else "compressed_per_component"
        logging.info(complexity_df[type].describe())
        plots_plus.gridlineplot(
            complexity_df,
            x="iteration",
            y=type,
            hue="version",
            axes="task",
            colorpalette=plots_plus.colors.LLM_COLOR_MAP,
            hue_order=version_order,
            ylim=(0, None),
            alpha=0.75,
            filepath=args.resultdir / f"{prefix}_complexity.png",  # type: ignore
        )
        bb_complexity_df = complexity_df[complexity_df["task"] == "Ball Balancing"]
        correlation = bb_complexity_df[["fitness_score_max", type]].corr("spearman")  # type: ignore
        logging.info(
            f"Ball Balancing: {prefix} Reward length - fitness score correlation (spearman): {correlation}"
        )
        plots_plus.scatterplot(bb_complexity_df, type, "fitness_score_max", hue="version", ylim=(-20, None), alpha=0.75, filepath=args.resultdir / f"bb_fitness_{prefix}_complexity.png")  # type: ignore

        fl_complexity_df = complexity_df[complexity_df["task"] == "Forward Locomotion"]
        correlation = fl_complexity_df[["fitness_score_max", type]].corr("spearman")  # type: ignore
        logging.info(
            f"Forward Locomotion: {prefix} Reward length - fitness score correlation (spearman): {correlation}"
        )
        plots_plus.scatterplot(fl_complexity_df, type, "fitness_score_max", hue="version", ylim=(-20, None), alpha=0.75, filepath=args.resultdir / f"fl_fitness_{prefix}_complexity.png")  # type: ignore

        plots_plus.gridscatterplot(complexity_df, type, "fitness_score_max", hue="version", axes="task", colorpalette=plots_plus.colors.LLM_COLOR_MAP, hue_order=version_order, ylim=(-20, None), alpha=0.75, filepath=args.resultdir / f"fitness_{prefix}_complexity.png")  # type: ignore
    return complexity_df


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
    rewards_df["task"] = stats_df["task"]

    resultdir = args.resultdir if args.resultdir else args.runs[0]
    resultdir.mkdir(exist_ok=True)

    analyze_reward_complexity(args, stats_df, rewards_df)

    if args.resultdir:
        with open(args.resultdir / "command.txt", "w") as f:
            f.write(" ".join([sys.executable, *sys.argv]))
