"""\
In this file I'll gather some of the implementation details that
I don't want to see the notebook.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import GridSearchCV
from IPython.display import display


def print_lib_versions(*libs):
    """Prints the versions of the libraries passed."""
    for lib in libs:
        print(f"{lib.__name__ + ':':12} {lib.__version__}")


def print_unique_column_entries(df: pd.DataFrame):
    """Prints the unique entries in each column."""
    print(f"Unique values for " f"{', '.join([str(col) for col in [*df]])}" ":")
    for col in df:
        unique_items = df[col].unique()
        print(f"- {col} ({len(unique_items)}): {*unique_items,}")


def describe(df: pd.DataFrame) -> None:
    """A nicer describe."""
    # fmt: off
    display(
        df
        .describe()
        .T
        .style.bar(subset=["mean"], cmap="viridis")
        .background_gradient(subset=["std", "50%", "max"], cmap="viridis")
    )
    # fmt: on


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary encoding
    df["sex"].replace({"female": 0, "male": 1}, inplace=True)
    df["smoker"].replace({"no": 0, "yes": 1}, inplace=True)

    # One-hot encoding
    region_dummies = pd.get_dummies(df["region"], dtype=int, prefix="region")
    df = pd.concat([df, region_dummies], axis=1)
    df.drop("region", axis=1, inplace=True)

    return df


def plot_charges_distribution(charges: pd.Series):
    """plots the 'charges' column."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        charges,
        kde=True,
        stat="density",
        kde_kws=dict(cut=0),
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
        bins=40,
        ax=ax,
        palette="viridis",
    )

    # Show the mode
    kdeline = ax.lines[0]
    xs, ys = kdeline.get_data()
    max_idx = np.argmax(ys)
    ax.vlines(
        xs[max_idx],
        0,
        ys[max_idx],
        linestyles="--",
        linewidth=2,
        color="#fde725",
    )

    # Show the mean value vertical line
    mean_val_x = charges.mean()
    plt.axvline(x=mean_val_x, linestyle="--", linewidth=2, color="#21918c")

    # Update the x ticks
    ax.set_xticks([0, xs[max_idx], 10_000, mean_val_x, *range(20_000, 60_001, 10_000)])
    ax.set_xticklabels(
        [
            "0",
            f"{xs[max_idx] / 1_000:.2f}k",
            "10k",
            f"{mean_val_x / 1_000:.2f}k",
            "20k",
            "30k",
            "40k",
            "50k",
            "60k",
        ]
    )

    plt.tight_layout()
    plt.xlabel("Charge")
    plt.show()


def plot_best_models(best_models: list[sklearn.base.BaseEstimator]):
    """Plots the best score of various models."""

    # Extract R^2 scores and best parameters from the results
    model_names = list(best_models.keys())
    r2_scores = [result[2] for result in best_models.values()]

    # Create a horizontal bar plot using Seaborn and sort the data
    data = list(zip(model_names, r2_scores))
    data.sort(key=lambda x: x[1])  # Sort by R^2 score

    sorted_model_names, sorted_r2_scores = zip(*data)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=sorted_r2_scores, y=sorted_model_names, palette="viridis")

    # Add R^2 scores in the middle of each bar
    for i, score in enumerate(sorted_r2_scores):
        barplot.text(
            score / 2,
            i,
            f"{score:.4f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

    plt.xlabel(r"$R^2$ Score")
    plt.ylabel("Model")
    plt.title(r"Comparison of $R^2$ Scores for Different Models (Sorted)")
    plt.xlim(0, 1)
    plt.show()


def is_running_on_cloud() -> bool:
    """Checks if the code is being run on a cloud environment like
    Google cloud or Kaggle"""
    environ = os.environ
    cloud_provider_envs = ["COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"]

    for cpe in cloud_provider_envs:
        if cpe in environ:
            return True

    return False


def compute_best_models(
    models, param_grids, X_train, y_train, cv=5, verbose=4
) -> tuple[dict, list[str]]:
    """Compute best models using GridSearchCV"""
    best_models: dict[str, tuple[sklearn.base.BaseEstimator, dict, float]] = {}
    performance_log: list[str] = []

    for name, model in models.items():
        param_grid = param_grids.get(name, {})

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", verbose=verbose)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        best_models[name] = (best_model, best_params, best_score)
        performance_log.append(
            f"{name}: Best Parameters - {best_params}, Best R^2 Score: {best_score:.5f}"
        )

    return best_models, performance_log


def plot_feature_importance(best_model, feature_names) -> None:
    """Plots the feature importance, given the best model
    and name of the all of the features."""
    importances = best_model.feature_importances_

    # Create a DataFrame to display feature names and their importance scores
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_df,
        palette="viridis",
    )

    ha: str
    imp: float
    color: str
    for i, importance in enumerate(-np.sort(-importances)):
        if importance > 0.1:
            ha = "center"
            imp = importance / 2
            color = "white"
        else:
            ha = "left"
            imp = importance + 0.005  # 0.005: padding
            color = "black"

        plt.text(
            imp,
            i,
            f"{importance:.4f}",
            ha=ha,
            va="center",
            color=color,
            fontsize=10,
        )

    plt.title("Gradient Boosting Feature Importance")
    plt.show()
