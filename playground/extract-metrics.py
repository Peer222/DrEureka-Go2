from pathlib import Path
from dataclasses import dataclass
import tyro
import re
from typing import List
import json
import pandas as pd
import numpy as np


@dataclass
class Args:
    eureka_run_dir: Path
    version: str = "no version"


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def construct_run_log(stdout_str):
    run_log = {}
    lines: List[str] = stdout_str.split("\n")
    for line in lines:
        if line.startswith("│") and line.endswith("│"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "train/episode/rew success/mean":
                key = "fitness_score"
            elif key == "timesteps" or key == "iterations":
                key = key
            elif "train/episode/rew" in key:
                key = key.split("/")[2]
            elif key == "train/episode/episode length/mean":
                key = "episode_length"
            elif "loss" in key:
                key = key.split("/")[0].split("mean ")[-1]

            run_log[key] = run_log.get(key, []) + [float(val)]
    if "fitness_score" not in run_log.keys():
        return None
    return run_log


def format_metrics(run_log):
    # Add reward components log to the feedback
    metrics = {}
    for metric in sorted(run_log.keys()):
        if metric not in ["timesteps", "iterations"]:
            if "fitness_score" == metric:
                metrics["fitness_score"] = run_log[metric]
            elif "episode_length" == metric:
                metrics["episode_length"] = run_log[metric]
            elif "total" in metric:
                metrics["total"] = run_log[metric]
            elif "rew" in metric:
                rew_name = metric.split("rew ")[-1]
                metrics[rew_name] = run_log[metric]
            elif "loss" in metric:
                # losses should not be included in llm feedback
                metrics[metric] = run_log[metric]
    return metrics


def get_iteration_and_sample_idx(log_file: Path):
    regex = re.search(r"iteration-(\d+)_sample-(\d+)", log_file.stem)
    # regex = re.search(r"env_iter(\d+)_response(\d+)", log_file.stem)  # for gpt-oss runs
    iteration, sample = -1, -1
    if regex:
        iteration = int(regex.group(1))
        sample = int(regex.group(2))
    return iteration, sample


def sorting_key(log_file: Path):
    iteration, sample = get_iteration_and_sample_idx(log_file)
    return iteration * 1000 + sample


def extract_metrics(run_dir: Path, version: str):
    log_files = list((run_dir / "logs").glob("iteration-*_sample-*.*"))
    # log_files = list((run_dir / "logs").glob("env_iter*_response*.*"))  # for gpt-oss runs
    log_files.sort(key=sorting_key)
    iteration_metrics = {}
    stats = []
    for log_file in log_files:
        print(log_file)
        iteration, sample = get_iteration_and_sample_idx(log_file)
        print(iteration, sample)
        if iteration not in iteration_metrics.keys():
            iteration_metrics[iteration] = []
        with open(log_file, "r") as f:
            stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg == "":
                run_log = construct_run_log(stdout_str)
                if not run_log:
                    iteration_metrics[iteration].append({})
                    stats = add_failure_values(stats, iteration, sample)
                else:
                    metrics = format_metrics(run_log)
                    iteration_metrics[iteration].append(metrics)

                    reward_names = []
                    for m in metrics.keys():
                        if m not in ["total", "fitness_score", "episode_length"] and "loss" not in m:
                            reward_names.append(m)
                    stats.append({
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "thinking_tokens": 0,
                        "answer_tokens": 0,
                        "total_tokens": 0,
                        "execution": 1,
                        "fitness_score_max": max(metrics["fitness_score"]),
                        "fitness_score_mean": np.mean(metrics["fitness_score"]),
                        "fitness_score_min": min(metrics["fitness_score"]),
                        "episode_length": max(metrics["episode_length"]),
                        "reward_total_max": max(metrics["total"]),
                        "reward_total_mean": np.mean(metrics["total"]),
                        "reward_total_min": min(metrics["total"]),
                        "num_reward_functions": len(reward_names),
                        "reward_names": reward_names,

                        "iteration": iteration,
                        "sample": sample,
                    })
            else:
                iteration_metrics[iteration].append({})
                stats = add_failure_values(stats, iteration, sample)

    full_metrics = [v for v in iteration_metrics.values()]
    with open(run_dir / "extracted_metrics.json", "w") as f:
        json.dump(full_metrics, f)

    stats_df = pd.DataFrame(stats)
    stats_df["version"] = version
    stats_df.to_csv(run_dir / "extracted_stats.csv")

def add_failure_values(stats, iteration, sample):
    stats.append({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_tokens": 0,
        "answer_tokens": 0,
        "total_tokens": 0,
        "execution": 0,
        "fitness_score_max": 0,
        "fitness_score_mean": 0,
        "fitness_score_min": 0,
        "episode_length": 0,
        "reward_total_max": 0,
        "reward_total_mean": 0,
        "reward_total_min": 0,
        "num_reward_functions": 0,
        "reward_names": [],

        "iteration": iteration,
        "sample": sample,
    })
    return stats

if __name__ == "__main__":
    args = tyro.cli(Args)
    extract_metrics(args.eureka_run_dir, args.version)
