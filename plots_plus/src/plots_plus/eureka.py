import tyro
import ast
from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd
from pathlib import Path
import plots_plus


def create_plots(
    model: str,
    full_stats_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    graphics_dir: Path,
):
    full_stats_df["reward_names"] = full_stats_df["reward_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    execution_rate_df = plots_plus.utils.to_execution_rates(full_stats_df)
    execution_rate_df["version"] = model
    plots_plus.lineplot(
        execution_rate_df,
        x="iteration",
        y="execution_rate",
        hue="version",
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
        ylim=(-10, None),
        filepath=graphics_dir / "fitness_score_max.png",
    )
    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="reward_total_max",
        hue="version",
        filepath=graphics_dir / "reward_total_max.png",
    )
    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="episode_length",
        hue="version",
        ylim=(0, None),
        filepath=graphics_dir / "episode_length.png",
    )
    plots_plus.scatteredlineplot(
        full_success_stats_df,
        x="iteration",
        y="num_reward_functions",
        hue="version",
        ylim=(0, None),
        filepath=graphics_dir / "num_reward_functions.png",
    )
    num_rew_per_iter_groups = full_success_stats_df.groupby("iteration")
    obj = {"iteration": [], "reward_names": []}
    for j, group in num_rew_per_iter_groups:
        names = []
        for i, row in group.iterrows():
            names.extend(row["reward_names"])
        obj["reward_names"].append(len(pd.Series(names).drop_duplicates()))
        obj["iteration"].append(j)
    num_rew_per_iter_df = pd.DataFrame(obj)
    num_rew_per_iter_df["version"] = full_stats_df["version"].iloc[0]
    plots_plus.lineplot(
        num_rew_per_iter_df,
        x="iteration",
        y="reward_names",
        hue="version",
        colorpalette=plots_plus.colors.LLM_COLOR_MAP,
        ylim=(0, None),
        filepath=graphics_dir / "reward_names.png",
    )

    tokens = plots_plus.utils.rotate_df(
        full_success_stats_df,
        "iteration",
        [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "thinking_tokens",
            "answer_tokens",
        ],
        "tokens",
    )
    plots_plus.lineplot(
        tokens,
        x="iteration",
        y="tokens",
        hue="type",
        ylim=(0, None),
        hue_order=plots_plus.colors.TOKEN_ORDER,
        colorpalette=plots_plus.colors.TOKEN_COLOR_MAP,
        filepath=graphics_dir / "tokens.png",
    )
    if "video_critique_prompt_tokens" in full_success_stats_df.columns:
        mapping = {
            "video_critique_prompt_tokens": "prompt_tokens",
            "video_critique_completion_tokens": "completion_tokens",
            "video_critique_total_tokens": "total_tokens",
            "video_critique_thinking_tokens": "thinking_tokens",
            "video_critique_answer_tokens": "answer_tokens",
        }
        token_df: pd.DataFrame = full_success_stats_df[[*mapping.keys(), "iteration"]]  # type: ignore
        token_df = token_df.rename(mapping, axis=1)
        tokens = plots_plus.utils.rotate_df(
            token_df,
            "iteration",
            list(mapping.values()),
            "tokens",
        )
        plots_plus.lineplot(
            tokens,
            x="iteration",
            y="tokens",
            hue="type",
            ylim=(0, None),
            hue_order=plots_plus.colors.TOKEN_ORDER,
            colorpalette=plots_plus.colors.TOKEN_COLOR_MAP,
            filepath=graphics_dir / "video_critique_tokens.png",
        )

    losses_df: pd.DataFrame = metrics_df[metrics_df["metric_name"].str.match(".*loss")].rename({"metric_name": "loss"}, axis=1)  # type: ignore
    plots_plus.gridlineplot(
        losses_df,
        x="training_iteration",
        y="value",
        hue="loss",
        axes="iteration",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "losses_per_iter.png",
    )
    plots_plus.gridlineplot(
        losses_df,
        x="training_iteration",
        y="value",
        hue="iteration",
        axes="loss",
        colorpalette=plots_plus.colors.ITERATION_COLOR_MAP,
        filepath=graphics_dir / "losses_per_type.png",
    )

    reward_components_df: pd.DataFrame = metrics_df[~metrics_df["metric_name"].isin(["episode_length", "fitness_score", "total"] + list(losses_df["loss"].drop_duplicates()))].rename({"metric_name": "reward"}, axis=1)  # type: ignore
    plots_plus.gridlineplot(
        reward_components_df,
        x="training_iteration",
        y="value",
        hue="reward",
        axes="iteration",
        colorpalette=plots_plus.colors.REWARD_COLOR_MAP,
        filepath=graphics_dir / "rewards_per_iter.png",
    )
    plots_plus.gridlineplot(
        reward_components_df,
        x="training_iteration",
        y="value",
        hue="iteration",
        axes="reward",
        colorpalette=plots_plus.colors.ITERATION_COLOR_MAP,
        filepath=graphics_dir / "rewards_per_type.png",
    )

    rew_total_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == "total"].rename({"value": "total_reward"}, axis=1)  # type: ignore
    plots_plus.multilineplot(
        rew_total_df,
        x="training_iteration",
        y="total_reward",
        lines="sample",
        hue="iteration",
        colorpalette=plots_plus.colors.ITERATION_COLOR_MAP,
        filepath=graphics_dir / "total_reward.png",
    )
    fitness_score_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == "fitness_score"].rename({"value": "fitness_score"}, axis=1)  # type: ignore
    plots_plus.multilineplot(
        fitness_score_df,
        x="training_iteration",
        y="fitness_score",
        lines="sample",
        hue="iteration",
        ylim=(-10, None),
        colorpalette=plots_plus.colors.ITERATION_COLOR_MAP,
        filepath=graphics_dir / "fitness_score.png",
    )

    # correlations
    corr_methods: List[Literal["spearman", "kendall", "pearson"]] = [
        "spearman",
        "kendall",
        "pearson",
    ]
    rewards_df: pd.DataFrame = metrics_df[~metrics_df["metric_name"].isin(["episode_length", "fitness_score"] + list(losses_df["loss"].drop_duplicates()))].rename({"metric_name": "reward"}, axis=1)  # type: ignore
    base_metric = "fitness_score"
    base_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == base_metric]  # type: ignore
    for corr_method in corr_methods:
        correlations_df = plots_plus.utils.get_correlation_df(
            base_df, rewards_df, "reward", corr_method
        )
        plots_plus.gridlineplot(
            correlations_df,
            x="iteration",
            y=f"{corr_method}_correlation",
            hue="samples",
            axes="reward",
            colorpalette=plots_plus.colors.CORRELATION_COLOR_MAP,
            filepath=graphics_dir / f"rew_fitness_correlation_{corr_method}.png",
            ylim=(-1.1, 1.1),
            markers=True,
        )

    base_metric = "value loss"
    base_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == base_metric]  # type: ignore
    for corr_method in corr_methods:
        correlations_df = plots_plus.utils.get_correlation_df(
            base_df, rewards_df, "reward", corr_method
        )
        plots_plus.gridlineplot(
            correlations_df,
            x="iteration",
            y=f"{corr_method}_correlation",
            hue="samples",
            axes="reward",
            colorpalette=plots_plus.colors.CORRELATION_COLOR_MAP,
            filepath=graphics_dir / f"rew_loss_correlation_{corr_method}.png",
            ylim=(-1.1, 1.1),
            markers=True,
        )


__all__ = ["create_plots"]


def __dir__():
    return __all__


if __name__ == "__main__":

    @dataclass
    class Args:
        result_dir: Path
        """directory in which graphics are stored"""
        train_iterations: int
        """Number of iterations used for training of samples"""
        run_dir: Optional[Path] = None
        """Path to run directory"""
        statspath: Optional[Path] = None
        """Path to eureka statistics file"""
        metricspath: Optional[Path] = None
        """Path to eureka rewards/metrics file"""

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    assert args.run_dir or (args.statspath and args.metricspath)

    if args.statspath:
        full_stats_df = pd.read_csv(args.statspath, index_col=0)
    else:
        full_stats_df = pd.read_csv(args.run_dir / "stats.csv", index_col=0)  # type: ignore
    graphics_dir = args.result_dir
    if args.metricspath:
        metrics_df = plots_plus.utils.load_metric_series(
            args.metricspath, args.train_iterations
        )
    else:
        metrics_df = plots_plus.utils.load_metric_series(
            args.run_dir / "metrics.json", args.train_iterations  # type: ignore
        )

    create_plots(
        full_stats_df["version"].iloc[0].split("_")[0],
        full_stats_df,
        metrics_df,
        args.result_dir,
    )
