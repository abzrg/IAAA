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
from IPython.display import display
from sklearn.model_selection import GridSearchCV, train_test_split


def is_running_on_cloud() -> bool:
    """Checks if the code is being run on a cloud environment like
    Google cloud or Kaggle"""
    environ = os.environ
    cloud_provider_envs = ["COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"]

    for cpe in cloud_provider_envs:
        if cpe in environ:
            return True

    return False


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


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Process the data and split it into training and test examples."""
    y = df["Outcome"]
    X = df.drop("Outcome", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=True, random_state=42
    )

    return X_train, X_test, y_train, y_test


def plot_corr_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="viridis")
    plt.show()


def age_outcome_ct_table(df: pd.DataFrame) -> None:
    """Displays a cross-tab information between age and outcome
    for 4 different age groups.

    src: https://www.youtube.com/watch?v=YlEXc6Kwoqc (Data Every Day #060)
    """
    age_ct = pd.crosstab(pd.qcut(df["Age"], q=4), df["Outcome"])
    age_ct_avgs = age_ct[1] / (age_ct[0] + age_ct[1])

    age_ct = pd.concat([age_ct, age_ct_avgs], axis=1)
    age_ct.columns = ["Negative", "Positive", "% Positive"]

    display(age_ct.style.bar(subset=["% Positive"], cmap="viridis_r"))


def plot_kde_features(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=True)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    for i, column in enumerate(df.columns):
        sns.kdeplot(df[column], shade=True, ax=axes[i], palette="viridis")

    plt.tight_layout()
    plt.show()
