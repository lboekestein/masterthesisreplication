"""
Plotting functions for the thesis project.

Contains:
  - get_disaggregated_metrics: builds a flat DataFrame containing all predictions, actuals, and their squared error for a DynamicModelManager
  - plot_mse_distance: plots MSE against distance between train and test splits, grouped by test_start
"""

from typing import Optional

from src.dynamic import DynamicModelManager
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import math


### --------------------------###
# Function to get disaggregated metrics from a DynamicModelManager #
### --------------------------###

def get_disaggregated_metrics(manager: DynamicModelManager) -> pd.DataFrame:
    """
    Build a flat DataFrame containing all predictions, actuals, and their squared error.
    If the target of the manager is log transformed, calculates squared_log_error instead of squared_error.
    Each row represents one [country_id, month_id, train_split, test_split, step] combination

    Returns a DataFrame with columns:
        country_id, prediction, train_start, train_end, test_start, test_end, step, distance, month_id, ln_ged_sb_dep, squared_[log]_error
    """
    predictions = []
    actuals = manager.data[["country_id", "month_id", manager.target]]

    for pred in manager.predictions:
        pred_dict = pred.predictions.copy()
        pred_dict['train_start'] = pred.train_split.start_month
        pred_dict['train_end'] = pred.train_split.end_month
        pred_dict['test_start'] = pred.test_split.start_month
        pred_dict['test_end'] = pred.test_split.end_month
        pred_dict['step'] = pred.train_split.step
        pred_dict['distance'] = pred.distance_

        predictions.append(pred_dict)

    full_predictions = pd.concat(predictions, ignore_index=True)

    merged = full_predictions.merge(
        actuals,
        left_on=["target_month_id", "country_id"],
        right_on=["month_id", "country_id"],
        how="inner"
    ).drop(columns=["target_month_id"])

    if manager.target_log_transformed:
        error_type = "squared_log_error"
    else:
        error_type = "squared_error"

    merged[error_type] = (merged[manager.target] - merged[pred.prediction_col]) ** 2

    return merged


### --------------------------###
# Actual plotting functions
### --------------------------###

def plot_error_over_time(
        metrics: pd.DataFrame,
        step: int,
        future_only: bool = True,
        model_name: str = "",
        group_by: str = "test_start",
        legend: bool = True,
        save: bool = False
    ) -> None:
    """
    Plot error over time for a given model manager.

    Args:
        metrics: DataFrame containing the disaggregatedmetrics to plot
        step: Step size for plotting
        future_only: Whether to plot only future predictions (test splits with distance > 0)
        model_name: Name of the model for labeling the plot
        group_by: Whether to group lines by "test_start" or "train_start"
        legend: Whether to show legend on the plot
        save: Whether to save the plot as a PNG file in ../figs/
    """

    # check if metric is present in columns
    if "squared_log_error" in metrics.columns:
        metric = "squared_log_error"
        label = "MSLE"
    elif "squared_error" in metrics.columns:
        metric = "squared_error"
        label = "MSE"
    else:
        raise ValueError("Expected 'squared_log_error' or 'squared_error' column not found in metrics DataFrame.")

    # check if group_by is valid
    if group_by not in ["test_start", "train_start"]:
        raise ValueError(f"Invalid group_by: {group_by}. Options are 'test_start' or 'train_start'")

    # check if step is valid
    if step not in metrics["step"].unique():
        raise ValueError(f"Invalid step: {step}. Available steps are {metrics['step'].unique()}")


    # subset to one step for plotting
    filt_metrics = metrics[metrics["step"] == step]

    # calculate mean squared log error for each test_start, train_start combination
    grouped = filt_metrics.groupby(["test_start", "train_start"])[[metric, "distance"]].mean().reset_index()

    if future_only:
        grouped = grouped[grouped["distance"] > 0]

    plt.figure(figsize=(12, 6))
    for (group_strategy), group in grouped.groupby([group_by]):
        plt.plot(group["distance"], group[metric], label=f"{group_by}={group_strategy}", alpha=0.6)
    plt.xlabel("Distance between train and test split (months)")
    plt.ylabel(label)
    plt.title(f"{label} vs Distance between Train and Test Splits" + 
            f"\nGrouped by {group_by}, {model_name}" + f", step = {step}")
    if legend:
        plt.legend()
    if save:
        plt.savefig(f"../figs/{metric}_distance_plot_aggregated_{model_name}_{step}_by_{group_by}.png")
    plt.show()


def plot_aggregated_error_over_time(
        metrics: pd.DataFrame,
        step: int,
        future_only: bool = True,
        model_name: str = "",
        legend: bool = True,
        save: bool = False
    ) -> None:
    """
    Plot aggregated error over time for a given model manager, averaging across all train splits for each test split.

    Args:
        metrics: DataFrame containing the disaggregated metrics to plot
        step: Step size for plotting
        future_only: Whether to plot only future predictions (test splits with distance > 0)
        model_name: Name of the model for labeling the plot
        legend: Whether to show legend on the plot
        save: Whether to save the plot as a PNG file in ../figs/
    """

    # check if metric is present in columns
    if "squared_log_error" in metrics.columns:
        metric = "squared_log_error"
        label = "MSLE"
    elif "squared_error" in metrics.columns:
        metric = "squared_error"
        label = "MSE"
    else:
        raise ValueError("Expected 'squared_log_error' or 'squared_error' column not found in metrics DataFrame.")

    # check if step is valid
    if step not in metrics["step"].unique():
        raise ValueError(f"Invalid step: {step}. Available steps are {metrics['step'].unique()}")


    # subset to one step for plotting
    filt_metrics = metrics[metrics["step"] == step]

    # calculate mean squared log error for each test_start, train_start combination
    grouped = filt_metrics.groupby(["test_start", "train_start"])[[metric, "distance"]].mean().reset_index()

    if future_only:
        grouped = grouped[grouped["distance"] > 0]

    # compute mean and std of mse by distance
    mean_mse = grouped.groupby("distance")[metric].mean()
    std_mse = grouped.groupby("distance")[metric].std()

    # plot mean mse and std of mse against distance
    plt.figure(figsize=(12, 6))
    plt.plot(mean_mse.index, mean_mse.values, label=metric, color="black", linestyle ="--") #type: ignore
    plt.fill_between(mean_mse.index, mean_mse.values - std_mse.values, mean_mse.values + std_mse.values, alpha=0.2, color="gray", label="±1 std") #type: ignore
    plt.xlabel("Distance between train and test split (months)")
    plt.ylabel(label)
    plt.title(f"Aggregated {label} vs Distance between Train and Test Splits" + f"\n{model_name}, step = {step}")
    if legend:
        plt.legend()
    if save:
        plt.savefig(f"../figs/{metric}_distance_plot_aggregated_{model_name}_{step}.png")
    plt.show()


def calculate_skill_score(
        metrics: pd.DataFrame, 
        baseline: str="test_start", 
        metric: str = "squared_log_error",
        future: bool = True
    ) -> pd.DataFrame:
    """
    Calculate skill score for each test split compared to baseline model (closest train or test split).
    Skill Score = 1 - (model_error / baseline_error)

    Args:
        metrics: DataFrame containing columns [baseline], 'distance', metric 
        baseline: Whether to calculate skill score relative to "test_start" or "train_start"
        metric: Column name of the error metric to use for skill score calculation (e.g., "squared_log_error", "squared_error")
        future: Whether to calculate skill score forwards or backwards in time. 
            Defaults to True, meaning skill score is calculated relative to closest train/test split in the future. 
            If False, skill score is calculated relative to closest train/test split in the past.
    """

    if baseline not in ["test_start", "train_start"]:
        raise ValueError(f"Invalid baseline: {baseline}. Options are 'test_start' or 'train_start'")

    if future:
        # calculate skill score relative to lowest positive distance
        idx = metrics.loc[metrics["distance"] > 0].groupby(baseline)["distance"].idxmin()
    else:
         # calculate skill score relative to highest negative distance
        idx = metrics.loc[metrics["distance"] < 0].groupby(baseline)["distance"].idxmax()

    # create lookup table
    baseline_error = metrics.loc[idx].set_index(baseline)[metric]

    # map back to dataframe
    metrics[f"baseline_{metric}"] = metrics[baseline].map(baseline_error)

    # calculate skill score
    metrics[f"skill_score_{metric}"] = 1 - (metrics[metric] / metrics[f"baseline_{metric}"])

    return metrics


def plot_skill_score_over_time(
        metrics: pd.DataFrame,
        step: int,
        future: bool = True,
        model_name: str = "",
        group_by: str = "test_start",
        save = False
    ) -> None:
    """
    Plot skill score over time for a given model manager.

    Args:
        metrics: DataFrame containing the disaggregated metrics to plot
        step: Step size for plotting
        future: Whether to plot skill score for future predictions (if true), or past predictions (if false)
        model_name: Name of the model for labeling the plot
        group_by: Whether to group lines by "test_start" or "train_start" for skill score calculation
    """
    
    # check if metric is present in columns
    if "squared_log_error" in metrics.columns:
        metric = "squared_log_error"
        label = "MSLE"
    elif "squared_error" in metrics.columns:
        metric = "squared_error"
        label = "MSE"
    else:
        raise ValueError("Expected 'squared_log_error' or 'squared_error' column not found in metrics DataFrame.")

    # check if group_by is valid
    if group_by not in ["test_start", "train_start"]:
        raise ValueError(f"Invalid group_by: {group_by}. Options are 'test_start' or 'train_start'")

    # check if step is valid
    if step not in metrics["step"].unique():
        raise ValueError(f"Invalid step: {step}. Available steps are {metrics['step'].unique()}")

    # subset to one step for plotting
    filt_metrics = metrics[metrics["step"] == step]

    # calculate mean squared log error for each test_start, train_start combination
    grouped = filt_metrics.groupby(["test_start", "train_start"])[[metric, "distance"]].mean().reset_index()

    skill_score = calculate_skill_score(grouped, baseline=group_by, metric=metric, future = future) #type: ignore

    if future:
        skill_score = skill_score[skill_score["distance"] > 0]
    else:
        skill_score = skill_score[skill_score["distance"] < 0]

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=skill_score,
        x="distance",
        y=f"skill_score_{metric}",    # separate lines by test_start
        alpha=0.7,
        estimator="mean",
        errorbar="sd",
    )
    plt.xlabel("Distance between train and test split (months)")
    plt.axhline(0, linestyle="--", color="black")  # baseline reference
    plt.ylabel(f"Skill Score ({label})")
    plt.title(f"Skill Score {label} vs Distance between Train and Test Splits" + f"\nGrouped by {group_by}, {model_name}, step = {step}")
    if save:
        plt.savefig(f"../figs/{metric}_skill_score_plot_{model_name}_{step}_{group_by}.png")
    plt.show()


def plot_combined_error_over_time(
        managers: dict[str, DynamicModelManager],
        step: int,
        metric: str = "squared_log_error",
        filter_list: Optional[list[int]] = None,
        future_only: bool = True,
        colormap: Optional[dict[str, str]] = None,
        save: bool = False
    ) -> None:
    """
    Plot error over time for multiple models on the same plot for comparison.

    Args:
        managers: Dictionary mapping model names to their DynamicModelManager instances
        step: Step size for plotting
        metric: Metric to plot (e.g., "squared_log_error" or "squared_error")
        filter_list: Optional list of country_ids to exclude from the plot (e.g., for robustness checks)
        future_only: Whether to plot only future predictions (test splits with distance > 0)
        colormap: Optional dictionary mapping model names to colors for plotting
        save: Whether to save the plot as a PNG file in ../figs/
    """

    # check if metric is valid and set label
    if metric == "squared_log_error":
        label = "MSLE"
    elif metric == "squared_error":
        label = "MSE"
    else:
        raise ValueError("Unexpected metric column found in metrics DataFrame. Expected 'squared_log_error' or 'squared_error'.")

    # set up plot
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="ticks", context="paper")

    if colormap is not None:
        sns.set_palette("muted")

    # get metrics for each model and plot error over time
    for model_name in managers.keys():

        metrics = get_disaggregated_metrics(managers[model_name])

        # check if metric is present in columns
        if metric not in metrics.columns:
            raise ValueError("Expected 'squared_log_error' or 'squared_error' column not found in metrics DataFrame.")
        
        # check if step is valid for given model
        if step not in metrics["step"].unique():
            raise ValueError(f"Invalid step: {step} for model {model_name}. Available steps are {metrics['step'].unique()}")

        # subset to specified step
        metrics = metrics[metrics["step"] == step]

        # filter out countries if specified (e.g., for robustness checks)
        if filter_list:
            metrics = metrics[~metrics["country_id"].isin(filter_list)]
            if len(metrics) == 0:
                raise ValueError(f"After filtering, no data left to plot for model {model_name}. Check filter_list and country_id values in metrics DataFrame.")

        # calculate mean squared [log] error for each test_start, train_start combination
        grouped = metrics.groupby(["test_start", "train_start"])[[metric, "distance"]].mean().reset_index()

        if future_only:
            grouped = grouped[grouped["distance"] > 0]

        # plot mean mse and std of mse against distance
        sns.lineplot(
            data=grouped,
            x="distance",
            y=metric,   
            estimator="mean",
            errorbar=None,
            label= f"{model_name}",
            linewidth=2.5,
            alpha = 0.9,
            color = colormap[model_name] if colormap is not None and model_name in colormap else None
            )
        
    plt.xlabel("Distance between train and test split (months)")
    plt.ylabel(f"Mean {label}")
    plt.legend(
        loc="upper center",
        ncol=math.ceil(len(managers) / 2),
        bbox_to_anchor=(0.5, 1.10),
        frameon=False
    )
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f"../figs/{metric}_combined_over_time_{step}.png")
    plt.show()


def plot_combined_skill_score_over_time(
        managers: dict[str, DynamicModelManager],
        step: int,
        metric: str = "squared_log_error",
        filter_list: Optional[list[int]] = None,
        group_by: str = "test_start",
        future: bool = True,
        colormap: Optional[dict[str, str]] = None,
        save: bool = False
    ) -> None:
    """
    Plot skill score over time for multiple models on the same plot for comparison.

    Args:
        managers: Dictionary mapping model names to their DynamicModelManager instances
        step: Step size for plotting
        metric: Metric to calculate skill score on (e.g., "squared_log_error", "squared_error")
        filter_list: Optional list of country_ids to exclude from the plot (e.g., for robustness checks)
        group_by: Whether to calculate skill score relative to "test_start" or "train_start"
        future: Whether to calculate skill score forwards or backwards in time. 
            If true, skill score is calculated relative to closest train/test split in the future. 
            If false, skill score is calculated relative to closest train/test split in the past.
        colormap: Optional dictionary mapping model names to colors for plotting
        save: Whether to save the plot as a PNG file in ../figs/
    """

    # check if group_by is valid
    if group_by not in ["test_start", "train_start"]:
        raise ValueError(f"Invalid group_by: {group_by}. Options are 'test_start' or 'train_start'")

    # check if metric is valid and set label
    if metric == "squared_log_error":
        label = "MSLE"
    elif metric == "squared_error":
        label = "MSE"
    else:
        raise ValueError("Unexpected metric column found in metrics DataFrame. Expected 'squared_log_error' or 'squared_error'.")

    # set up plot
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="ticks", context="paper")

    if colormap is not None:
        sns.set_palette("muted")

    # get metrics for each model and plot error over time
    for model_name in managers.keys():

        metrics = get_disaggregated_metrics(managers[model_name])

        # check if metric is present in columns
        if metric not in metrics.columns:
            raise ValueError("Expected 'squared_log_error' or 'squared_error' column not found in metrics DataFrame.")
        
        # check if step is valid for given model
        if step not in metrics["step"].unique():
            raise ValueError(f"Invalid step: {step} for model {model_name}. Available steps are {metrics['step'].unique()}")

        # subset to specified step
        metrics = metrics[metrics["step"] == step]

        # filter out countries if specified (e.g., for robustness checks)
        if filter_list:
            metrics = metrics[~metrics["country_id"].isin(filter_list)]
            if len(metrics) == 0:
                raise ValueError(f"After filtering, no data left to plot for model {model_name}. Check filter_list and country_id values in metrics DataFrame.")

        # calculate mean squared [log] error for each test_start, train_start combination
        grouped = metrics.groupby(["test_start", "train_start"])[[metric, "distance"]].mean().reset_index()

        skill_score = calculate_skill_score(grouped, baseline=group_by, metric=metric, future = future) #type: ignore

        if future:
            skill_score = skill_score[skill_score["distance"] > 0]
        else:
            skill_score = skill_score[skill_score["distance"] < 0]

        sns.lineplot(
            data=skill_score,
            x="distance",
            y=f"skill_score_{metric}",    # separate lines by test_start
            estimator="mean",
            errorbar=("ci", 95),
            err_kws= {"alpha": 0.05},
            err_style="band",
            label= f"{model_name}",
            linewidth=2.5,
            alpha = 0.9,
            color = colormap[model_name] if colormap is not None and model_name in colormap else None
        )
        
    plt.xlabel("Distance between train and test split (months)")
    plt.axhline(0, linestyle="--", color="grey", linewidth=1.2)  # baseline reference
    plt.ylabel(f"Skill Score ({label})")
    plt.legend(
        loc="upper center",
        ncol=math.ceil(len(managers) / 2),
        bbox_to_anchor=(0.5, 1.10),
        frameon=False
    )
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(f"../figs/{metric}_combined_skill_score_over_time_{step}_{group_by}.png")
    plt.show()