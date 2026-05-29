from typing import List, Tuple, Literal, Union, Iterable, Dict, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


def clean_df_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalizes words and removes underscores of column labels

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with updated column labels
    """
    df.columns
    name_mapping = {}
    for label in df.columns:
        name_mapping[label] = clean_variable(label)
    return df.rename(name_mapping, axis=1)


def clean_variable(var_name: str) -> str:
    """Capitalizes words and removes underscores

    Args:
        var_name (str): unformatted variable/column name

    Returns:
        str: Formatted name
    """
    return var_name.replace("_", " ").title()


def to_execution_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Converts binary execution success criterion into percentage (per iteration)
    Args:
        df (pd.DataFrame): DataFrame with iteration and execution column

    Returns:
        pd.DataFrame: DataFrame with iteration and execution_rate columns
    """
    grouping_vars = ["iteration"]
    if "version" in df.columns:
        grouping_vars += ["version"]
    if "task" in df.columns:
        grouping_vars += ["task"]
    if "seed" in df.columns:
        grouping_vars += ["seed"]
    execution_rates = df.groupby(grouping_vars, as_index=False)["execution"].mean()
    return pd.DataFrame(execution_rates).rename({"execution": "execution_rate"}, axis=1)


def load_metric_series(filepath: Path, train_iterations: int) -> pd.DataFrame:
    """Loads and converts json reward/fitness series to pandas df

    Args:
        filepath (Path): Path to rewards.json
        train_iterations (int): Total number of training iterations

    Returns:
        pd.DataFrame: Dataframe with the following columns: []
    """
    with open(filepath, "r") as f:
        rewards = json.load(f)
        return convert_metric_series(rewards, train_iterations)


def rotate_df(
    df: pd.DataFrame, x: Union[str, List[str]], ys: Iterable[str], ylabel: str
) -> pd.DataFrame:
    """Rotates df und returns df with x, "type", and ylabel column
    Type column is cleaned with clean_variable()

    Args:
        df (pd.DataFrame): Stats DataFrame
        x (str): column name for x plotting value and eventually hues/styles
        ys (List[str]): Columns to stack onto each other
        ylabel (str): Label of new y column

    Returns:
        pd.DataFrame: DataFrame with columns: x, "type", ylabel
    """
    if isinstance(x, str):
        x = [x]
    new_df = pd.DataFrame()
    for y in ys:
        partial_df = df[[*x, y]].rename({y: ylabel}, axis=1)
        partial_df["type"] = clean_variable(y)
        new_df = pd.concat([new_df, partial_df])
    return new_df


def convert_metric_series(series: List, train_iterations: int) -> pd.DataFrame:
    """Converts reward series json data into dataframe

    Args:
        series (List): JSON stored data
        train_iterations (int): Number of training iterations per evaluation

    Returns:
        pd.DataFrame: Dataframe with columns: sample, iteration, metric_name, value, training_iteration
    """
    df_struct = {
        "sample": [],
        "iteration": [],
        "metric_name": [],
        "value": [],
        "training_iteration": [],
    }
    iteration_interval = None
    for i_idx, iteration in enumerate(series):
        for s_idx, sample in enumerate(iteration):
            for key, values in sample.items():
                if key == "training_iteration":
                    continue
                if not iteration_interval:
                    num_entries = len(values)
                    iteration_interval = list(
                        np.arange(0, train_iterations, train_iterations // num_entries)
                    )
                if "training_iteration" not in sample.keys():
                    df_struct["training_iteration"].extend(iteration_interval)
                else:
                    df_struct["training_iteration"].extend(sample["training_iteration"])

                df_struct["value"].extend(values)
                df_struct["iteration"].extend([i_idx for _ in values])
                df_struct["sample"].extend([s_idx for _ in values])
                df_struct["metric_name"].extend([key for _ in values])

    df = pd.DataFrame(df_struct).astype(
        {
            "sample": int,
            "iteration": int,
            "metric_name": str,
            "value": float,
            "training_iteration": int,
        }
    )
    return df


def axgrid(
    nrows: int,
    ncols: int,
    row_height: float = 5,
    col_width: float = 5,
    legend_height: float = 0.75,
    rel_hspace: float = 0.35,
    rel_wspace: float = 0.21,
) -> Tuple:
    """Creates a subplot grid with additional space on top reserved for a figure legend

    Args:
        nrows (int): Number of rows
        ncols (int): Number of columns
        row_height (float, optional): Row height. Defaults to 5.
        col_width (float, optional): Column width. Defaults to 5.
        legend_height (float, optional): Height of legend. Defaults to 0.75.
        rel_hspace (float, optional): Relative height reserved for subplot descriptions/title. Defaults to 0.35.

    Returns:
        Tuple: figure, plot axes, legend axis
    """
    fig = plt.figure(figsize=(ncols * col_width, legend_height + nrows * row_height))

    gs = mpl.gridspec.GridSpec(  # type: ignore
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=[legend_height] + [row_height] * nrows,
        hspace=rel_hspace,
        wspace=rel_wspace,
        figure=fig,
    )

    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    axs = [fig.add_subplot(gs[r + 1, c]) for r in range(nrows) for c in range(ncols)]
    return fig, axs, legend_ax


def get_correlation_df(
    base_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    column: str,
    method: Literal["pearson", "kendall", "spearman"],
) -> pd.DataFrame:
    """Computes iteration-wise correlations of all samples and only on best sample
    result dataframes have the following columns: {method}_correlation, iteration, {column}, samples
    samples column has options: all, best

    Args:
        base_df (pd.DataFrame): dataframe based on which all correlations are computed
        metrics_df (pd.DataFrame): dataframe with categories to compute correlations
        column (str): Name of category column (e.g. reward)
        method (Literal["pearson", "kendall", "spearman"]): Correlation computation method

    Returns:
        pd.DataFrame: Correlations
    """
    metrics = metrics_df.groupby(column, as_index=False)
    correlations = []

    for name, metric_df in metrics:
        base_iterations = base_df.groupby("iteration", as_index=False)
        for iteration, base_iteration_df in base_iterations:
            corr = compute_correlation(base_iteration_df, metric_df, method=method)
            correlations.append(
                {
                    f"{method}_correlation": corr,
                    "iteration": iteration,
                    column: name,
                    "samples": "all",
                }
            )

            best_idx = base_iteration_df["value"].idxmax()
            best_sample = base_df.loc[[best_idx]]["sample"].iloc[0]
            corr_best = compute_correlation(
                base_iteration_df[base_iteration_df["sample"] == best_sample],
                metric_df,
                method=method,
            )
            correlations.append(
                {
                    f"{method}_correlation": corr_best,
                    "iteration": iteration,
                    column: name,
                    "samples": "best",
                }
            )

    return pd.DataFrame(correlations)


def compute_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str = "value",
    on: Union[str, Iterable[str]] = ["iteration", "sample", "training_iteration"],
    method: Literal["pearson", "kendall", "spearman"] = "spearman",
) -> float:
    """Computes Correlation

    Args:
        left (pd.DataFrame): DataFrame
        right (pd.DataFrame): DataFrame
        column (str, optional): Column on which to perform correlation calculation. Defaults to "value".
        on (Union[str, Iterable[str]], optional): Based on which columns to perform join. Defaults to ["iteration", "sample", "training_iteration"].
        method (Literal["pearson", "kendall", "spearman"], optional): Type of correlation. Defaults to "pearson".

    Returns:
        float: Correlation between both series
    """
    merged = pd.merge(left, right, on=on, suffixes=["_base", "_reward"])
    return merged[f"{column}_base"].corr(merged[f"{column}_reward"], method=method)


def get_limits(
    df: pd.DataFrame,
    col: str,
    method: Union[Tuple[Union[float, None], Union[float, None]], str],
    pad_factor: float = 1.6,
) -> Union[Tuple[Union[float, None], Union[float, None]], Tuple[None]]:
    if isinstance(method, tuple):
        return method
    if method not in ["minmax", "auto"]:
        return (None,)
    if method == "minmax":
        max_spread = max(df[col].abs())
        return min(df[col]) - max_spread / 50, max(df[col]) + max_spread / 50
    elif method == "auto":
        percentiles = df[col].where(df[col] != 0).describe(percentiles=[0.025, 0.975])
        # catch only zero values
        if percentiles["count"] == 0:
            return -1, 1
        return max(pad_factor * percentiles["2.5%"], percentiles["min"]), min(
            pad_factor * percentiles["97.5%"], percentiles["max"]
        )
    raise NotImplementedError(f"{method} not implemented for limits")


def strip_palette(
    palette: Union[List, mpl.colors.Colormap, None], df: pd.DataFrame, hue: Optional[str]
) -> Union[List, mpl.colors.Colormap, None]:
    if hue is None or palette is None:
        return None
    if isinstance(palette, mpl.colors.Colormap):
        return palette
    num = min(len(palette), len(df[hue].drop_duplicates()))
    return [palette[i] for i in range(num)]


def multi_legend(ax, colorpalette: Union[List, mpl.colors.Colormap, None]):
    """Expects axis with scatterplot legend in form of Version, ..., Task, ...

    Args:
        ax (axes): Axis
        colorpalette (Union[List, mpl.colors.Colormap]): Colorpalette
    """
    if colorpalette is None:
        return
    scatter_handles, labels = ax.get_legend_handles_labels()
    line_handles = []
    i = 0
    for label in labels:
        if label == "Version":
            i = 0
            line_handles.append(Line2D([], [], color="white", label=label))
            continue
        if label == "Task":
            i = 0
            line_handles.append(Line2D([], [], color="white", label=label))
            # 2 tasks
            line_handles.append(Line2D([], [], linestyle="solid", label=label))
            line_handles.append(Line2D([], [], linestyle="dashed", label=label))
            break

        if isinstance(colorpalette, mpl.colors.Colormap):
            color = colorpalette(i * (1 / len(labels)))
        else:
            color = colorpalette[i]
        line = Line2D([], [], color=color, linestyle="solid", label=label)
        line_handles.append(line)
        i += 1

    # legend containing line and marker symbols
    combined_handles = []
    for h1, h2 in zip(line_handles, scatter_handles):
        combined_handles.append((h1, h2))
    ax.legend(
        handles=combined_handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},  # type: ignore
        title=None,
    )


# control package visibility
__all__ = [
    "axgrid",
    "rotate_df",
    "convert_metric_series",
    "load_metric_series",
    "to_execution_rates",
    "clean_variable",
    "clean_df_labels",
    "get_correlation_df",
    "compute_correlation",
    "get_limits",
    "strip_palette",
    "multi_legend",
]


def __dir__():
    return __all__
