from typing import List, Tuple, Optional, Union, Literal
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

from .utils import *
from .colors import *

FIGSIZE = (7,5) # (10, 7)


def plot(filepath: Optional[Path] = None) -> None:
    """Shows figure or saves figure if filepath is specified

    Args:
        filepath (Optional[Path], optional): Path of saved file. Defaults to None.
    """
    if not filepath:
        plt.show()
    else:
        if not filepath.parent.is_dir():
            filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=filepath, dpi=300, bbox_inches="tight")
        plt.close()


def scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: Union[List, mpl.colors.Colormap, None] = LLM_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    xlim: Union[Tuple[Union[float, Union[float, None]], float], str] = "minmax",
    ylim: Union[Tuple[Union[float, None], Union[float, None]], str] = "auto",
    hue_order: Optional[List[str]] = None,
    style: Optional[str] = None,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)
    if style:
        style = clean_variable(style)

    colorpalette = strip_palette(colorpalette, df, hue)

    plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        df, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order, style=style
    )

    ax.set_xlim(*get_limits(df, x, xlim))  # type: ignore
    ax.set_ylim(*get_limits(df, y, ylim))  # type: ignore

    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.SUBTLE_GREY)
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title:
        plt.title(title)
    if hue:
        plt.legend()
    plot(filepath)


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: Union[List, mpl.colors.Colormap, None] = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    xlim: Union[Tuple[Union[float, Union[float, None]], float], str] = "minmax",
    ylim: Union[Tuple[Union[float, None], Union[float, None]], str] = "auto",
    hue_order: Optional[List[str]] = None,
    style: Optional[str] = None,
    alpha: float = 1.0,
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)
    if style:
        style = clean_variable(style)

    colorpalette = strip_palette(colorpalette, df, hue)

    plt.figure(figsize=FIGSIZE)
    ax = sns.lineplot(
        df, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order, style=style
    )

    for line in ax.lines:
        line.set_alpha(alpha)

    ax.set_xlim(*get_limits(df, x, xlim))  # type: ignore
    ax.set_ylim(*get_limits(df, y, ylim))  # type: ignore

    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.SUBTLE_GREY)
    if x == "Iteration":
        ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    if title:
        plt.title(title)
    if hue:
        plt.legend()
    plot(filepath)


def scatteredlineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: Union[List, mpl.colors.Colormap, None] = LLM_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    xlim: Union[Tuple[Union[float, None], Union[float, None]], str] = "minmax",
    ylim: Union[Tuple[Union[float, None], Union[float, None]], str] = "auto",
    hue_order: Optional[List[str]] = None,
    style: Optional[str] = None,
    alpha: float = 1.0,
    errorbar: Optional[Tuple[str, float]] = ("ci", 95),
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)
    if style:
        style = clean_variable(style)

    colorpalette = strip_palette(colorpalette, df, hue)

    plt.figure(figsize=FIGSIZE)
    ax = sns.lineplot(
        df,
        x=x,
        y=y,
        hue=hue,
        palette=colorpalette,
        legend=False,
        hue_order=hue_order,
        style=style,
        #alpha=alpha,
        errorbar=errorbar,
    )
    ax = sns.scatterplot(
        df,
        x=x,
        y=y,
        hue=hue,
        palette=colorpalette,
        hue_order=hue_order,
        style=style,
        alpha=alpha,
    )
    if style:
        # TODO only works for hue=Version, style=Task (with two tasks)
        multi_legend(ax, colorpalette)

    ax.set_xlim(*get_limits(df, x, xlim))  # type: ignore
    ax.set_ylim(*get_limits(df, y, ylim))  # type: ignore

    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.SUBTLE_GREY)
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
    colorpalette: Union[List, mpl.colors.Colormap, None] = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    xlim: Union[Tuple[Union[float, None], Union[float, None]], str] = "minmax",
    ylim: Union[Tuple[Union[float, None], Union[float, None]], str] = "auto",
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

    colorpalette = strip_palette(colorpalette, df, hue)

    ax = None
    plt.figure(figsize=FIGSIZE)
    for v in df[lines].drop_duplicates():
        filtered_df: pd.DataFrame = df[df[lines] == v]  # type: ignore
        ax = sns.lineplot(
            filtered_df, x=x, y=y, hue=hue, hue_order=hue_order, palette=colorpalette
        )

    if ax is None:
        raise Exception("No axes created")
    for line in ax.lines:
        line.set_alpha(alpha)

    ax.set_xlim(*get_limits(df, x, xlim))  # type: ignore
    ax.set_ylim(*get_limits(df, y, ylim))  # type: ignore

    ax.tick_params(direction="in", length=0)
    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.SUBTLE_GREY)
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
    colorpalette: Union[List, mpl.colors.Colormap, None] = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    xlim: Union[Tuple[Union[float, None], Union[float, None]], str] = "minmax",
    ylim: Union[Tuple[Union[float, None], Union[float, None]], str] = "auto",
    markers: Optional[Union[bool, Literal["max"]]] = None,
    hue_order: Optional[List[str]] = None,
    alpha: float = 1.0,
    errorbar: Optional[Tuple[str, float]] = ("ci", 95),
):
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    hue = clean_variable(hue)
    axes = clean_variable(axes)

    colorpalette = strip_palette(colorpalette, df, hue)

    groups = df.groupby(axes)

    max_cols = 5  # change if names are too long
    nrows = int(np.ceil(len(groups) / max_cols))
    ncols = int(min(len(groups), max_cols))
    num_labels = len(df[hue].drop_duplicates())
    legend_cols = 2 * ncols if num_labels > 2 * ncols else num_labels
    fig, axs, legend_ax = axgrid(
        nrows, ncols, legend_height=np.ceil(num_labels / legend_cols) * 0.3
    )

    if not hue_order:
        names: List[str] = list(df[hue].drop_duplicates().sort_values())
        if len(names) > 1:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "multiline", colorpalette, N=len(names)
            )
            colorpalette = [cmap(i) for i in np.linspace(0, 1, len(names))]
        hue_order = names

    for ax, (name, group) in zip(axs, groups):
        name = clean_variable(str(name))
        sns.lineplot(
            group, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order, ax=ax, alpha=alpha, errorbar=errorbar
        )

        ax.set_title(name)
        ax.set_xlim(*get_limits(df, x, xlim))
        ax.set_ylim(*get_limits(df, y, ylim))

        if markers == "max":
            by = [x, hue]
            if "Seed" in group.columns:
                by += ["Seed"]
            max_group: pd.DataFrame = group.loc[group.groupby(by)[y].idxmax()]  # type: ignore
            sns.scatterplot(
                max_group, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order, ax=ax, alpha=alpha
            )
        elif markers:
            sns.scatterplot(
                group, x=x, y=y, hue=hue, palette=colorpalette, hue_order=hue_order, ax=ax, alpha=alpha
            )

        ax.tick_params(direction="in", length=0)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True, right=True, top=True)
        ax.grid(True, color=Color.SUBTLE_GREY)
        if x == "Iteration":
            ax.set_xticks(np.arange(0, len(df[x].drop_duplicates()), 1))

    # build single figure legend
    legend_items = OrderedDict()
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            #legend_items[clean_variable(l)] = h  # deduplicate by label
            legend_items[l] = h  # deduplicate by label
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
        handletextpad=0.1,
    )

    # Remove unused subplots
    for ax in axs[len(groups) :]:
        fig.delaxes(ax)

    if title:
        plt.title(title)
    plot(filepath)


def clusterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    colorpalette: Union[List, mpl.colors.Colormap, None] = ITERATION_COLOR_MAP,
    filepath: Optional[Path] = None,
    title: Optional[str] = None,
    hue_order: Optional[List[str]] = None,
    style: Optional[str] = None,
    size: Optional[str] = None,
    alpha: float = 1.0,
):
    import adjustText
    df = clean_df_labels(df)
    x = clean_variable(x)
    y = clean_variable(y)
    if hue:
        hue = clean_variable(hue)
    if style:
        style = clean_variable(style)
    if size:
        size = clean_variable(size)
    colorpalette = strip_palette(colorpalette, df, hue)

    plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        df, x=x, y=y, hue=hue, hue_order=hue_order, palette=colorpalette, style=style, size=size, alpha=alpha
    )

    ax.set_axisbelow(True)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.grid(True, color=Color.SUBTLE_GREY)

    # remove uninformative ticks and labels
    ax.tick_params(colors="white", which="both")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if x == "X" and y == "Y":
        ax.set_xlabel("")
        ax.set_ylabel("")

    if "Text" in df.columns:
        texts = []
        target_x = []
        target_y = []
        for i, row in df.iterrows():
            color = Color.BLACK
            if "Count" in df.columns and row["Count"] <= 2:
                color = Color.GREY
            texts.append(ax.text(row[x] + 0.01, row[y], str(row["Text"]), fontsize=8, color=color))
            target_x.append(row[x])
            target_y.append(row[y])
        adjustText.adjust_text(
            texts, target_x=target_x, target_y=target_y, force_pull=(0.03, 0.03)
        )

    if title:
        plt.title(title)
    if hue:
        plt.legend()
    plot(filepath)
