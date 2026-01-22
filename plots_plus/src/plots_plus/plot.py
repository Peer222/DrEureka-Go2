from typing import List, Tuple, Optional, Literal
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass
import tyro

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

from .utils import *
from .colors import *


def plot(filepath: Optional[Path] = None) -> None:
    """Shows figure or saves figure if filepath is specified

    Args:
        plt (_type_): Matplotlib plot/ figure
        filepath (Optional[Path], optional): Path of saved file. Defaults to None.
    """
    if not filepath:
        plt.show()
    else:
        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=filepath, dpi=300, bbox_inches="tight")
        plt.figure(clear=True)
        plt.close()


def scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: List = LLM_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    hue_order: Optional[List[str]] = None,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        df, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order
    )

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted
    ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title:
        plt.title(title)
    plt.legend()
    plot(filepath)


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: List = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    hue_order: Optional[List[str]] = None,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(df, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted TODO
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title:
        plt.title(title)
    plt.legend()
    plot(filepath)


def scatteredlineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: List = LLM_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    hue_order: Optional[List[str]] = None,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(
        df, x=x, y=y, hue=hue, palette=colorpalette, legend=False, hue_order=hue_order
    )
    ax = sns.scatterplot(
        df, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order
    )

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title:
        plt.title(title)
    plot(filepath)


def multilineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    lines: str,
    hue: Optional[str] = None,
    colorpalette: List = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    alpha: float = 0.7,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    lines = clean_variable(lines)
    hue_order = []
    if hue:
        hue = clean_variable(hue)
        hue_order = df[hue].drop_duplicates().sort_values()

    ax = None
    plt.figure(figsize=(10, 7))
    for v in df[lines].drop_duplicates():
        filtered_df = df[df[lines] == v]
        ax = sns.lineplot(
            filtered_df, x=x, y=y, hue=hue, hue_order=hue_order, palette=colorpalette
        )

    if ax is None:
        raise Exception("No axes created")
    for line in ax.lines:
        line.set_alpha(alpha)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    # Legend without duplicated entries
    legend_items = OrderedDict()
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        legend_items[clean_variable(l)] = h  # deduplicates by label
    ax.legend(
        legend_items.values(),
        legend_items.keys(),
        title=hue,
        frameon=False,
    )

    if title:
        plt.title(title)
    plot(filepath)


def gridlineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    axes: str,
    colorpalette: List = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    hue = clean_variable(hue)
    axes = clean_variable(axes)

    groups = df.groupby(axes)

    max_cols = 5
    nrows = int(np.ceil(len(groups) / max_cols))
    ncols = int(min(len(groups), max_cols))
    num_labels = len(df[hue].drop_duplicates())
    legend_cols = np.ceil(num_labels / 2) if num_labels > 5 else num_labels
    fig, axs, legend_ax = axgrid(
        nrows, ncols, legend_height=np.ceil(num_labels / legend_cols) * 0.3
    )

    names: List[str] = list(df[hue].drop_duplicates().sort_values())
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "multiline", colorpalette, N=len(names)
    )
    colorpalette = [cmap(i) for i in np.linspace(0, 1, len(names))]

    for ax, (name, group) in zip(axs, groups):
        name = clean_variable(str(name))
        sns.lineplot(
            group, x=x, y=y, hue=hue, palette=colorpalette, hue_order=names, ax=ax
        )

        ax.set_title(name)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(min(df[x]), max(df[x]) + 0.1)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(min(df[y]), max(df[y]))
        ax.tick_params(direction="in", length=0)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True, right=True, top=True)
        ax.grid(True, color=Color.LIGHT_GREY)
        if x == "Iteration":
            ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    # build single figure legend
    legend_items = OrderedDict()
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_items[clean_variable(l)] = h  # deduplicate by label
        axlegend = ax.get_legend()
        if axlegend:
            axlegend.remove()
    legend_ax.legend(
        legend_items.values(),
        legend_items.keys(),
        loc="upper center",
        ncol=legend_cols,
        title=hue,
        frameon=False,
    )

    # Remove unused subplots
    for ax in axs[len(groups) :]:
        fig.delaxes(ax)

    if title:
        plt.title(title)
    plot(filepath)


if __name__ == "__main__":

    @dataclass
    class Args:
        statspath: Path
        """Path to eureka statistics file"""
        metricspath: Path
        """Path to eureka rewards/metrics file"""
        result_dir: Path
        """directory in which graphics are stored"""
        train_iterations: int = 300
        """Number of iterations used for training of samples"""

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    full_stats = pd.read_csv(args.statspath)
    graphics_dir = args.result_dir

    execution_rate_df = to_execution_rates(full_stats)
    execution_rate_df["version"] = full_stats["version"].iloc[0]
    lineplot(
        execution_rate_df,
        x="iteration",
        y="execution_rate",
        hue="version",
        colorpalette=LLM_COLOR_MAP,
        ylim=(-0.1, 1.1),
        filepath=graphics_dir / "execution_rates.png",
    )

    # use only successful evaluations
    full_stats: pd.DataFrame = full_stats[full_stats["execution"] == 1]  # type: ignore (vscode bug)
    scatteredlineplot(
        full_stats,
        x="iteration",
        y="fitness_score_max",
        hue="version",
        filepath=graphics_dir / "fitness_score_max.png",
    )
    scatteredlineplot(
        full_stats,
        x="iteration",
        y="reward_total_max",
        hue="version",
        filepath=graphics_dir / "reward_total_max.png",
    )
    scatteredlineplot(
        full_stats,
        x="iteration",
        y="episode_length",
        hue="version",
        filepath=graphics_dir / "episode_length.png",
    )
    scatteredlineplot(
        full_stats,
        x="iteration",
        y="num_reward_functions",
        hue="version",
        ylim=(0, 12),
        filepath=graphics_dir / "num_reward_functions.png",
    )

    tokens = rotate_df(
        full_stats,
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
    lineplot(
        tokens,
        x="iteration",
        y="tokens",
        hue="type",
        hue_order=TOKEN_ORDER,
        colorpalette=TOKEN_COLOR_MAP,
        filepath=graphics_dir / "tokens.png",
    )

    metrics_df = load_metric_series(args.metricspath, args.train_iterations)

    losses_df: pd.DataFrame = metrics_df[metrics_df["metric_name"].str.match(".*loss")].rename({"metric_name": "loss"}, axis=1)  # type: ignore
    gridlineplot(
        losses_df,
        x="training_iteration",
        y="value",
        hue="loss",
        axes="iteration",
        colorpalette=REWARD_COLOR_MAP,
        filepath=graphics_dir / "losses_per_iter.png",
    )
    gridlineplot(
        losses_df,
        x="training_iteration",
        y="value",
        hue="iteration",
        axes="loss",
        colorpalette=ITERATION_COLOR_MAP,
        filepath=graphics_dir / "losses_per_type.png",
    )

    reward_components_df: pd.DataFrame = metrics_df[~metrics_df["metric_name"].isin(["episode_length", "fitness_score", "total"] + list(losses_df["loss"].drop_duplicates()))].rename({"metric_name": "reward"}, axis=1)  # type: ignore
    gridlineplot(
        reward_components_df,
        x="training_iteration",
        y="value",
        hue="reward",
        axes="iteration",
        colorpalette=REWARD_COLOR_MAP,
        filepath=graphics_dir / "rewards_per_iter.png",
    )
    gridlineplot(
        reward_components_df,
        x="training_iteration",
        y="value",
        hue="iteration",
        axes="reward",
        colorpalette=ITERATION_COLOR_MAP,
        filepath=graphics_dir / "rewards_per_type.png",
    )

    rew_total_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == "total"].rename({"value": "total_reward"}, axis=1)  # type: ignore
    multilineplot(
        rew_total_df,
        x="training_iteration",
        y="total_reward",
        lines="sample",
        hue="iteration",
        colorpalette=ITERATION_COLOR_MAP,
        filepath=graphics_dir / "total_reward.png",
    )
    fitness_score_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == "fitness_score"].rename({"value": "fitness_score"}, axis=1)  # type: ignore
    multilineplot(
        fitness_score_df,
        x="training_iteration",
        y="fitness_score",
        lines="sample",
        hue="iteration",
        colorpalette=ITERATION_COLOR_MAP,
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
        correlations_df = get_correlation_df(base_df, rewards_df, "reward", corr_method)
        gridlineplot(
            correlations_df,
            x="iteration",
            y=f"{corr_method}_correlation",
            hue="samples",
            axes="reward",
            colorpalette=CORRELATION_COLOR_MAP,
            filepath=graphics_dir / f"rew_fitness_correlation_{corr_method}.png",
            ylim=(-1.1, 1.1),
        )

    base_metric = "value loss"
    base_df: pd.DataFrame = metrics_df[metrics_df["metric_name"] == base_metric]  # type: ignore
    for corr_method in corr_methods:
        correlations_df = get_correlation_df(base_df, rewards_df, "reward", corr_method)
        gridlineplot(
            correlations_df,
            x="iteration",
            y=f"{corr_method}_correlation",
            hue="samples",
            axes="reward",
            colorpalette=CORRELATION_COLOR_MAP,
            filepath=graphics_dir / f"rew_loss_correlation_{corr_method}.png",
            ylim=(-1.1, 1.1),
        )
