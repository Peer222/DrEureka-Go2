from typing import Union, Optional, Tuple, List, Literal, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import re
import sys
import json
from dataclasses import dataclass
import openai
import logging
import time
from pydantic import BaseModel
from enum import Enum


@dataclass
class Args:
    runs: List[Path]
    """Path to stats.csv files from eureka runs (multiple seeds)"""


def analyze_reward_evolution(
    stats_df: pd.DataFrame, prefix: str = ""
):
    deletions = {}
    creations = {}
    prev_iteration_df = None
    for iteration, iteration_df in stats_df.groupby("iteration"):
        deletions[iteration] = 0
        creations[iteration] = 0
        for index, sample in iteration_df.iterrows():
            parent_names = []
            if "ancestors" in stats_df.columns:
                if len(sample["ancestors"]) and isinstance(sample["ancestors"][0], int):
                    sample["ancestors"] = [sample["ancestors"]]

                for ancestor in sample["ancestors"]:
                    parent_names += stats_df[
                        (stats_df["iteration"] == ancestor[0])
                        & (stats_df["sample"] == ancestor[1])
                    ]["reward_names"].item()
            elif isinstance(prev_iteration_df, pd.DataFrame):
                best_idx = prev_iteration_df["fitness_score_max"].idxmax()
                parent_names = prev_iteration_df.loc[best_idx, "reward_names"]
            
            parent_names = set(parent_names)
            reward_names = set(sample["reward_names"])
            deletions[iteration] += len(parent_names - reward_names)
            creations[iteration] += len(reward_names - parent_names)
            
        prev_iteration_df = iteration_df
        
    print(f"{prefix}: Deletions: {deletions}, Creations: {creations}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import tyro

    args = tyro.cli(Args)

    for run_dir in args.runs:
        stats_df = pd.read_csv(run_dir / "stats.csv")

        match = re.search(".*/([^_]+)_.*", stats_df["version"].iloc[0])
        if match is None:
            raise Exception(
                f"Unknown model version found: {stats_df['version'].iloc[0]}"
            )
        version = f"{match.group(1)}-{stats_df['seed'].iloc[0]}"
        stats_df["version"] = match.group(1)
        stats_df.reset_index(inplace=True)

        if "ancestors" in stats_df.columns:
            stats_df["ancestors"] = stats_df["ancestors"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        stats_df["reward_names"] = stats_df["reward_names"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        if "GW" in run_dir.stem:
            version = "BB " + version
        elif "FL" in run_dir.stem:
            version = "FL " + version
        else:
            raise NotImplementedError(
                f"{run_dir.stem} needs to include either FL or GW"
            )

        analyze_reward_evolution(stats_df, prefix=version)
