from typing import List, Tuple, Optional
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
        plt.savefig(fname=filepath, dpi=300, bbox_inches='tight')
        plt.figure(clear=True)
        plt.close()

def scatterplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, colorpalette: List = LLM_COLOR_MAP, filepath: Optional[Path]=None, title: Optional[str]=None, ylim: Optional[Tuple[float, float]] = None, xlim: Optional[Tuple[float, float]] = None):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue: hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(df, x=x, y=y, hue=hue, palette=colorpalette)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted
    ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title: plt.title(title)
    plt.legend()
    plot(filepath)


def lineplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, colorpalette: List = ITERATION_COLOR_MAP, filepath: Optional[Path]=None, title: Optional[str]=None, ylim: Optional[Tuple[float, float]] = None, xlim: Optional[Tuple[float, float]] = None):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue: hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(df, x=x, y=y, hue=hue, palette=colorpalette)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted TODO
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title: plt.title(title)
    plt.legend()
    plot(filepath)


def scatteredlineplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, colorpalette: List = LLM_COLOR_MAP, filepath: Optional[Path]=None, title: Optional[str]=None, ylim: Optional[Tuple[float, float]] = None, xlim: Optional[Tuple[float, float]] = None):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue: hue = clean_variable(hue)

    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(df, x=x, y=y, hue=hue, palette=colorpalette, legend=False)
    ax = sns.scatterplot(df, x=x, y=y, hue=hue, palette=colorpalette)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.LIGHT_GREY)
    # for eval iterations. Otherwise has to be adapted
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title: plt.title(title)
    plot(filepath)


def multilineplot(df: pd.DataFrame, x: str, y: str, lines: str, hue: Optional[str] = None, colorpalette: List = ITERATION_COLOR_MAP, filepath: Optional[Path]=None, title: Optional[str]=None, ylim: Optional[Tuple[float, float]] = None, xlim: Optional[Tuple[float, float]] = None, alpha: float = 0.7):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    lines = clean_variable(lines)
    hue_order = []
    if hue: 
        hue = clean_variable(hue)
        hue_order = df[hue].drop_duplicates()

    ax = None
    plt.figure(figsize=(10, 7))
    for v in df[lines].drop_duplicates():
        filtered_df = df[df[lines] == v]
        ax = sns.lineplot(filtered_df, x=x, y=y, hue=hue, hue_order=hue_order, palette=colorpalette)

    if ax is None:
        raise Exception("No axes created")
    for line in ax.lines:
        line.set_alpha(alpha)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
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

    if title: plt.title(title)
    plot(filepath)


def gridlineplot(df: pd.DataFrame, x: str, y: str, hue: str, axes: str, colorpalette: List = ITERATION_COLOR_MAP, filepath: Optional[Path]=None, title: Optional[str]=None, ylim: Optional[Tuple[float, float]] = None, xlim: Optional[Tuple[float, float]] = None):
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
    fig, axs, legend_ax = axgrid(nrows, ncols, legend_height=np.ceil(num_labels / legend_cols) * 0.3)

    names: List[str]  = list(df[hue].drop_duplicates())
    cmap = mpl.colors.LinearSegmentedColormap.from_list("multiline", colorpalette, N=len(names))
    colorpalette = [cmap(i) for i in np.linspace(0, 1, len(names))]

    for ax, (name, group) in zip(axs, groups):
        name = clean_variable(str(name))
        sns.lineplot(group, x=x, y=y, hue=hue, palette=colorpalette, hue_order=names, ax=ax)

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
        if axlegend: axlegend.remove()
    legend_ax.legend(
        legend_items.values(),
        legend_items.keys(),
        loc="upper center",
        ncol=legend_cols,
        title=hue,
        frameon=False,
    )

    # Remove unused subplots
    for ax in axs[len(groups):]:
        fig.delaxes(ax)

    if title: plt.title(title)
    plot(filepath)


if __name__ == "__main__":
    @dataclass
    class Args():
        statspath: Path
        """Path to eureka statistics file"""
        rewardspath: Path
        """Path to eureka rewards/metrics file"""
        train_iterations: int = 500
        """Number of iterations used for training of samples"""
        result_dir: Path = Path(__file__)

    args = tyro.cli(Args)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    eureka_stats = pd.read_csv(args.statspath)
    # TODO autofill presentable version
    eureka_stats["version"] = "Test"

    ### generate eureka stats plots
    scatterplot(eureka_stats, x="iteration", y="fitness_score_max", hue="version", filepath=args.result_dir / "fitness_score_max_scatter.png")
    scatteredlineplot(eureka_stats, x="iteration", y="fitness_score_max", hue="version", filepath=args.result_dir / "fitness_score_max_scatterline.png")
    lineplot(eureka_stats, x="iteration", y="fitness_score_max", hue="version", colorpalette=LLM_COLOR_MAP, filepath=args.result_dir / "fitness_score_max_line.png")

    execution_rates = to_execution_rates(eureka_stats)
    execution_rates["version"] = "Test"
    lineplot(execution_rates, x="iteration", y="execution_rate", hue="version", colorpalette=LLM_COLOR_MAP, ylim=(0, 1), filepath=args.result_dir / "execution_rates.png")

    scatteredlineplot(eureka_stats, x="iteration", y="fitness_score_max", hue="version", filepath=args.result_dir / "fitness_score_max.png")
    scatteredlineplot(eureka_stats, x="iteration", y="reward_total_max", hue="version", filepath=args.result_dir / "reward_total_max.png")
    scatteredlineplot(eureka_stats, x="iteration", y="episode_length", hue="version", filepath=args.result_dir / "episode_length.png")
    scatteredlineplot(eureka_stats, x="iteration", y="num_reward_functions", hue="version", filepath=args.result_dir / "num_reward_functions.png")

    tokens = rotate_df(eureka_stats, "iteration", ["prompt_tokens", "completion_tokens", "total_tokens"], "tokens")
    lineplot(tokens, x="iteration", y="tokens", hue="type", colorpalette=LLM_COLOR_MAP, filepath=args.result_dir / "tokens.png")

    eureka_metrics = load_metric_series(args.rewardspath, args.train_iterations)
    eureka_losses: pd.DataFrame = eureka_metrics[eureka_metrics["metric_name"].str.match("loss")].rename({"metric_name": "loss"}, axis=1)  # type: ignore
    loss_names = list(eureka_losses["loss"].drop_duplicates())
    eureka_rewards: pd.DataFrame = eureka_metrics[~eureka_metrics["metric_name"].isin(["episode_length", "fitness_score", "total"] + loss_names)].rename({"metric_name": "reward"}, axis=1)  # type: ignore

    base_metric = "height"
    base: pd.DataFrame = eureka_metrics[eureka_metrics["metric_name"] == base_metric]  # type: ignore
    for corr_method in ["spearman", "kendall", "pearson"]:
        correlations = get_correlation_df(base, eureka_rewards, "reward", corr_method)  #type: ignore
        gridlineplot(correlations, x="iteration", y=f"{corr_method}_correlation", hue="samples", axes="reward", colorpalette=CORRELATION_COLOR_MAP, filepath=args.result_dir / f"correlation_{corr_method}.png", ylim=(-1.1, 1.1))

    gridlineplot(eureka_rewards, x="training_iteration", y="value", hue="reward", axes="iteration", colorpalette=REWARD_COLOR_MAP, filepath=args.result_dir / "rewards_per_iter.png")
    gridlineplot(eureka_rewards, x="training_iteration", y="value", hue="iteration", axes="reward", colorpalette=ITERATION_COLOR_MAP, filepath=args.result_dir / "rewards_per_type.png")
    orientation_rewards: pd.DataFrame = eureka_rewards[eureka_rewards["reward"] == "orientation"].rename({"value": "orientation"}, axis=1)  # type: ignore
    multilineplot(orientation_rewards, x="training_iteration", y="orientation", lines="sample", hue="iteration", colorpalette=ITERATION_COLOR_MAP, filepath=args.result_dir / "orientation.png")
