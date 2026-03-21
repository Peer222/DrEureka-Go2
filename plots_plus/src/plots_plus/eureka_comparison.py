import tyro
import ast
from dataclasses import dataclass
from typing import List, Literal, Optional
import re

import pandas as pd
from pathlib import Path
import plots_plus


def create_plots(
    full_stats_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    graphics_dir: Path,
):
    full_stats_df["reward_names"] = full_stats_df["reward_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    execution_rate_df = plots_plus.utils.to_execution_rates(full_stats_df)
    plots_plus.lineplot(
        execution_rate_df,
        x="iteration",
        y="execution_rate",
        hue="version",
        style="task",
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        ylim=(-0.1, 1.1),
        filepath=graphics_dir / "execution_rates.png",
    )

    # use only successful evaluations
    full_success_stats_df: pd.DataFrame = full_stats_df[full_stats_df["execution"] == 1]  # type: ignore (vscode bug)
    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="fitness_score_max",
        hue="version",
        style="task",
        ylim=(-10, None),
        alpha=0.6,
        errorbar=None,
        filepath=graphics_dir / "fitness_score_max.png",
    )
    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="reward_total_max",
        hue="version",
        style="task",
        alpha=0.6,
        errorbar=None,
        filepath=graphics_dir / "reward_total_max.png",
    )

    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="num_reward_functions",
        hue="version",
        style="task",
        ylim=(0, None),
        alpha=0.6,
        errorbar=None,
        filepath=graphics_dir / "num_reward_functions.png",
    )

    num_rew_per_iter_groups = full_success_stats_df.groupby(
        ["iteration", "version", "task"]
    )
    obj = {"iteration": [], "reward_names": [], "version": [], "task": []}
    for (iteration, version, task), group in num_rew_per_iter_groups:
        names = []
        for i, row in group.iterrows():
            names.extend(row["reward_names"])
        obj["reward_names"].append(len(pd.Series(names).drop_duplicates()))
        obj["iteration"].append(iteration)
        obj["version"].append(version)
        obj["task"].append(task)
    num_rew_per_iter_df = pd.DataFrame(obj)
    plots_plus.lineplot(
        num_rew_per_iter_df,
        x="iteration",
        y="reward_names",
        hue="version",
        style="task",
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        ylim=(0, None),
        alpha=0.6,
        filepath=graphics_dir / "reward_names.png",
    )

    plots_plus.lineplot(
        full_success_stats_df,
        x="iteration",
        y="completion_tokens",
        hue="version",
        style="task",
        ylim=(0, None),
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        alpha=0.6,
        filepath=graphics_dir / "completion_tokens.png",
    )
    plots_plus.lineplot(
        full_success_stats_df,
        x="iteration",
        y="thinking_tokens",
        hue="version",
        style="task",
        ylim=(0, None),
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        alpha=0.6,
        filepath=graphics_dir / "thinking_tokens.png",
    )
    plots_plus.lineplot(
        full_success_stats_df,
        x="iteration",
        y="answer_tokens",
        hue="version",
        style="task",
        ylim=(0, None),
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        alpha=0.6,
        filepath=graphics_dir / "answer_tokens.png",
    )

    if "video_critique_prompt_tokens" in full_success_stats_df.columns:
        mapping = {
            "video_critique_prompt_tokens": "prompt_tokens",
            "video_critique_completion_tokens": "completion_tokens",
            "video_critique_total_tokens": "total_tokens",
            "video_critique_thinking_tokens": "thinking_tokens",
            "video_critique_answer_tokens": "answer_tokens",
        }
        token_df: pd.DataFrame = full_success_stats_df[[*mapping.keys(), "iteration", "version", "task"]]  # type: ignore
        token_df = token_df.rename(mapping, axis=1)
        plots_plus.lineplot(
            full_success_stats_df,
            x="iteration",
            y="completion_tokens",
            hue="version",
            style="task",
            ylim=(0, None),
            colorpalette=plots_plus.colors.LLM_COLOR_MAP,
            alpha=0.6,
            filepath=graphics_dir / "video_critique_completion_tokens.png",
        )
        plots_plus.lineplot(
            full_success_stats_df,
            x="iteration",
            y="thinking_tokens",
            hue="version",
            style="task",
            ylim=(0, None),
            colorpalette=plots_plus.colors.LLM_COLOR_MAP,
            alpha=0.6,
            filepath=graphics_dir / "video_critique_thinking_tokens.png",
        )
        plots_plus.lineplot(
            full_success_stats_df,
            x="iteration",
            y="answer_tokens",
            hue="version",
            style="task",
            ylim=(0, None),
            colorpalette=plots_plus.colors.LLM_COLOR_MAP,
            alpha=0.6,
            filepath=graphics_dir / "video_critique_answer_tokens.png",
        )

    # correlations
    corr_methods: List[Literal["spearman", "kendall", "pearson"]] = [
        "spearman",
        "kendall",
        "pearson",
    ]
    rewards_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == "total"].rename({"metric_name": "reward"}, axis=1)  # type: ignore
    base_metric = "fitness_score"
    base_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == base_metric]  # type: ignore
    for corr_method in corr_methods:
        grouped_reward = rewards_df.groupby(["version", "task"])
        grouped_base = base_df.groupby(["version", "task"])
        full_correlations_df = pd.DataFrame()
        for (index, base_group), (_, reward_group) in zip(grouped_base, grouped_reward):
            correlations_df = plots_plus.utils.get_correlation_df(
                base_group, reward_group, "reward", corr_method
            )
            correlations_df["version"] = index[0]
            correlations_df["task"] = index[1]
            full_correlations_df = pd.concat([full_correlations_df, correlations_df])
        plots_plus.lineplot(
            full_correlations_df,
            x="iteration",
            y=f"{corr_method}_correlation",
            hue="version",
            style="task",
            ylim=(-1.1, 1.1),
            colorpalette=plots_plus.colors.LLM_COLOR_MAP,
            alpha=0.6,
            filepath=graphics_dir / f"rew_fitness_correlation_{corr_method}.png",
        )


__all__ = ["create_plots"]


def __dir__():
    return __all__


if __name__ == "__main__":

    @dataclass
    class Args:
        run_dirs: List[Path]
        """Path to run directory"""
        result_dir: Path
        """directory in which graphics are stored"""
        train_iterations: int  # TODO relative scale or short for forward locomotion
        """Number of iterations used for training of samples"""

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    graphics_dir = args.result_dir

    all_stats_df = pd.DataFrame()
    all_metrics_df = pd.DataFrame()
    for run_dir in args.run_dirs:
        stats_df = pd.read_csv(run_dir / "stats.csv", index_col=0)  # type: ignore
        metrics_df = plots_plus.utils.load_metric_series(
            run_dir / "metrics.json", args.train_iterations  # type: ignore
        )
        match = re.search(".*/([^_]+)_.*", stats_df["version"].iloc[0])
        if match is None:
            raise Exception(
                f"Unknown model version found: {stats_df['version'].iloc[0]}"
            )

        stats_df["version"] = match.group(1)
        metrics_df["version"] = match.group(1)
        if "GW" in run_dir.stem:
            stats_df["task"] = "Balancing"
            metrics_df["task"] = "Balancing"
        elif "FL" in run_dir.stem:
            stats_df["task"] = "Locomotion"
            metrics_df["task"] = "Locomotion"
        else:
            raise NotImplementedError(
                f"{run_dir.stem} needs to include either FL or GW"
            )

        all_stats_df = pd.concat([all_stats_df, stats_df])
        all_metrics_df = pd.concat([all_metrics_df, metrics_df])

    create_plots(
        all_stats_df,
        all_metrics_df,
        args.result_dir,
    )
