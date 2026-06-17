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
    model: str = "Qwen/Qwen3.6-27B-FP8" #"openai/gpt-oss-20b"  # "Qwen/Qwen3-14B-AWQ"
    """Name of the huggingface embedding model name"""
    models_dir: Path = Path("/bigwork/nhwpduep/master_thesis/models/")
    """Path to model root directory"""
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
        with open(reward_file, "r") as f:
            content = f.read()

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


def get_reward_evolution(reward: str, parents: List[str], model: str):
    class OperationType(str, Enum):
        deletion = "deletion"
        creation = "creation"
        selection = "selection"
        mutation = "mutation"
        rescaling = "rescaling"
        crossover = "crossover"

    class RewardComponentEvolution(BaseModel):
        reward_name: str
        description: str
        parents: List[int]
        operation: OperationType

    class RewardEvolution(BaseModel):
        reward_components: List[RewardComponentEvolution]

    initial_system: str = """
        You are a reinforcement learning researcher. Your task is to classify the evolution of the reward functions. The possible operations and their explanations are as follows: 
            creation: Reward function name is new and not in one of the parents (only applicable if REWARD FUNCTION NAME IS COMPLETELY NEW and not present in parent)  
            deletion: Every parent component not mapped to an offspring component receives a deletion label.  
            selection: Reward component is reused from parent WITHOUT ANY CHANGES  
            mutation: Reward component is reused from parent but with MODIFIED COMPUTATION (do not assign for simple change of an external multiplicative coefficient)  
            rescaling: Reward component is reused but the SCALING FACTOR is changed (Rescaling applies only when the reward computation is identical and only an external multiplicative coefficient changes.)  
            crossover: Reward component is a COMBINATION of both parent components, connecting formulation ideas (only applicable if multiple parents shown)  
            
        To make it more clear an example is given with the correctly annotated operations:
        Parent 0:
            def _reward_height(self):
                target_height = 2.0 * env.ball_radius
                reward = torch.clamp(env.base_pos[:, 2] / target_height, max=1.0)
                return 1.0 * reward

            def _reward_orientation(self):
                reward = 1.0 - torch.norm(env.projected_gravity[:, :2], dim=1)
                return 1.0 * reward

            def _reward_lin_vel(self):
                reward = -torch.sum(torch.square(env.base_lin_vel), dim=1)
                return 0.05 * reward

            def _reward_ang_vel(self):
                reward = -torch.sum(torch.square(env.base_ang_vel), dim=1)
                return 0.03 * reward

            def _reward_action_rate(self):
                reward = -torch.sum(torch.square(env.actions - env.last_actions), dim=1)
                return 0.005 * reward

        Offspring:
            def _reward_height(self):
                target_height = 2.0 * env.ball_radius
                reward = torch.clamp(env.base_pos[:, 2] / target_height, max=1.0)
                return 1.0 * reward

            def _reward_orientation_penalty(self):
                reward = 1.0 - torch.norm(env.projected_gravity[:, :2], dim=1)
                return 0.03 * reward

            def _reward_lin_vel(self):
                reward = -torch.sum(torch.abs(env.base_lin_vel), dim=1)
                return 0.8 * reward

            def _reward_ang_vel(self):
                reward = -torch.sum(torch.square(env.base_ang_vel), dim=1) * 10
                return 0.03 * reward

            def _reward_action_rate(self):
                reward = -torch.sum(torch.exp(torch.abs(env.actions - env.last_actions) / 0.5)))
                return 0.005 * reward

            def compute_fitness_score(self):
                return 1

        Classifications for each reward component:
            _reward_height -> { reward_name: height, operation: selection, parents: [0], description: Reused the reward component without any changes in computation, name or scaling }
            _reward_orientation -> { reward_name: orientation, operation: deletion, parents: [0], description: The reward component was removed }
            _reward_orientation_penalty -> { reward_name: orientation_penalty, operation: creation, parents: [], description: A newly introduced reward that calculates ... }
            _reward_lin_vel -> { reward_name: lin_vel, operation: mutation, parents: [0], description: The reward component now uses the absolute base_lin_vel instead of squaring it. Additionally the reward was rescaled from 0.05 to 0.8 }
            _reward_ang_vel -> { reward_name: ang_vel, operation: rescaling, parents: [0], description: The reward component was rescaled by a factor of 10 }
            _reward_action_rate -> { reward_name: action_rate, operation: mutation, parents: [0], description: The reward component was reformulated and is now based on the exponentiated action differences instead of the squared}
    """

    instruction = f"{len(parents)} parent and the offspring reward configuration are provided below. Your goal is to classify the operations that produced the offspring. Therefore, you should classifiy each individual reward component, present in parents and offspring, based on the provided taxonomy. You should include the reward component name (without _reward_), a description that summarizes the operation/changes in detail in a single sentence, the parent index (or indices) and the operation in the requested format. You need to strictly follow the taxonomy and its instructions!\n\n"
    rewards = ""
    for i, parent in enumerate(parents):
        rewards += f"Parent {i}:  \n{parent}  \n"
    rewards += f"Offspring:  \n{reward}"
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": instruction + rewards},
    ]

    logging.debug(messages)
    # return RewardEvolution(reward_components=[RewardComponentEvolution(reward_name="test", parents=[0], operation=OperationType.selection), RewardComponentEvolution(reward_name="test2", parents=[0, 1], operation=OperationType.crossover)]).model_dump()

    openai.api_key = "..."
    vllm_host = f"http://0.0.0.0:8000"
    openai.api_base = f"{vllm_host}/v1"

    response = None
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reward-evolution-analysis",
                        "schema": RewardEvolution.model_json_schema(),
                    },
                },
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            break
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()

    logging.info(response["choices"][0]["message"]["content"])  # type: ignore
    answer = response["choices"][0]["message"]["content"]  # type: ignore
    return json.loads(answer)


def analyze_reward_evolution(
    args: Args, stats_df: pd.DataFrame, rewards_df: pd.DataFrame, prefix: str = ""
):
    evolution_data = []
    for index, sample in stats_df.iterrows():
        version = sample["version"]
        seed = sample["seed"]
        i_idx = sample["iteration"]
        s_idx = sample["sample"]

        offspring_reward = rewards_df.iloc[index]["content"]
        parent_rewards = []
        if len(sample["ancestors"]) and isinstance(sample["ancestors"][0], int):
            sample["ancestors"] = [sample["ancestors"]]
        parent_score = 0
        for ancestor in sample["ancestors"]:
            parent_rewards.append(
                rewards_df[
                    (rewards_df["seed"] == seed)
                    & (rewards_df["iteration"] == ancestor[0])
                    & (rewards_df["sample"] == ancestor[1])
                ]["content"]
            )
            parent_score += stats_df[
                (rewards_df["seed"] == seed)
                & (rewards_df["iteration"] == ancestor[0])
                & (rewards_df["sample"] == ancestor[1])
            ]["fitness_score_max"].item()

        reward_evolution = get_reward_evolution(
            offspring_reward, parent_rewards, f"{args.models_dir}/{args.model}"
        )
        logging.debug(f"{reward_evolution=}")
        fitness_score_diff = sample["fitness_score_max"] - parent_score / max(
            len(sample["ancestors"]), 1
        )
        for component in reward_evolution["reward_components"]:
            evolution_data.append(
                {
                    "version": version,
                    "seed": seed,
                    "iteration": i_idx,
                    "sample": s_idx,
                    "fitness_score_max": sample["fitness_score_max"],
                    "fitness_score_diff": fitness_score_diff,
                    "operation": component["operation"],
                    "reward_name": component["reward_name"],
                    "parents": component["parents"],
                    "description": component["description"],
                }
            )

    evolution_df = pd.DataFrame(evolution_data)
    evolution_df["fitness_score_max"] = evolution_df["fitness_score_max"].round(2)
    evolution_df["fitness_score_diff"] = evolution_df["fitness_score_diff"].round(2)
    evolution_df.to_csv(args.resultdir / f"{prefix}reward_evolution.csv")  # type: ignore
    logging.info(evolution_df["operation"].value_counts())
    return evolution_df


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    import tyro

    args = tyro.cli(Args)
    assert (
        len(args.runs) > 1 and args.resultdir or len(args.runs) == 1
    ), "If multiple runs are shown, a result dir must be specified"

    resultdir = args.resultdir if args.resultdir else args.runs[0]
    resultdir.mkdir(exist_ok=True)

    for run_dir in args.runs:
        stats_df = pd.read_csv(run_dir / "stats.csv")

        match = re.search(".*/([^_]+)_.*", stats_df["version"].iloc[0])
        if match is None:
            raise Exception(
                f"Unknown model version found: {stats_df['version'].iloc[0]}"
            )
        version = f"{match.group(1)}-{stats_df['seed'].iloc[0]}-"
        stats_df["version"] = match.group(1)
        stats_df.reset_index(inplace=True)
        stats_df["ancestors"] = stats_df["ancestors"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        rewards_df = get_rewards(run_dir / "rewards")
        rewards_df.reset_index(inplace=True)
        rewards_df["seed"] = stats_df["seed"]
        rewards_df["version"] = stats_df["version"]

        analyze_reward_evolution(args, stats_df, rewards_df, prefix=version)

    if args.resultdir:
        with open(args.resultdir / "command.txt", "w") as f:
            f.write(" ".join([sys.executable, *sys.argv]))
