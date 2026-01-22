from typing import List, Tuple, Literal, Union, Iterable, Dict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    (only applicable to single version df / versions need to be added afterwards)
    Args:
        df (pd.DataFrame): DataFrame with iteration and execution column

    Returns:
        pd.DataFrame: DataFrame with iteration and execution_rate columns
    """
    execution_rates = df.groupby("iteration", as_index=False)["execution"].mean()
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


def rotate_df(df: pd.DataFrame, x: str, ys: List[str], ylabel: str) -> pd.DataFrame:
    """Rotates df und returns df with x, "type", and ylabel column
    Type column is cleaned with clean_variable()

    Args:
        df (pd.DataFrame): Stats DataFrame
        x (str): column name for x plotting value
        ys (List[str]): Columns to stack onto each other
        ylabel (str): Label of new y column

    Returns:
        pd.DataFrame: DataFrame with columns: x, "type", ylabel
    """
    new_df = pd.DataFrame()
    for y in ys:
        partial_df = df[[x, y]].rename({y: ylabel}, axis=1)
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
                num_entries = len(values)
                if not iteration_interval:
                    iteration_interval = list(
                        np.arange(0, train_iterations, train_iterations // num_entries)
                    )

                df_struct["value"].extend(values)
                df_struct["training_iteration"].extend(iteration_interval)
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
        float: _description_
    """
    merged = pd.merge(left, right, on=on, suffixes=["_base", "_reward"])
    return merged[f"{column}_base"].corr(merged[f"{column}_reward"], method=method)


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
]


def __dir__():
    return __all__
