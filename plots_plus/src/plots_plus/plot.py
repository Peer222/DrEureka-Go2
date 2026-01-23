from typing import List, Tuple, Optional
from collections import OrderedDict
from pathlib import Path

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
