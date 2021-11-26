import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import Dataset


def get_x_label(dataset: Dataset, i: int) -> str:
    return "\n".join([
        dataset.feature_names[i],
        "Mean: " + ", ".join([f"{x}: {dataset.means[dataset.feature_names[i]][x]}" for x in dataset.means[dataset.feature_names[i]]]),
        "SD: " + ", ".join([f"{x}: {dataset.stds[dataset.feature_names[i]][x]}" for x in dataset.stds[dataset.feature_names[i]]])
    ])


def plot_distribution(dataset: Dataset) -> None:
    fig, axes = plt.subplots(15, 2, figsize=(20, 50))
    ax = axes.ravel()

    for i in range(29):
        sns.kdeplot(data=dataset.df, x=dataset.feature_names[i], hue="QC", fill=True, common_norm=False, bw_adjust=1, alpha=.5, ax=ax[i])
        ax[i].set_xlabel(get_x_label(dataset, i))
    
    fig.tight_layout(h_pad=2)
