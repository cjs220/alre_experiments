import pandas as pd
from matplotlib import pyplot as plt


def plot_line_graph_with_errors(mean: pd.DataFrame, stderr: pd.DataFrame, ax=None, alpha=0.3, **kwargs):
    ax = mean.plot(ax=ax)
    ax.set_prop_cycle(None)
    for col in stderr.columns:
        ax.fill_between(
            mean.index.values,
            mean[col].values - stderr[col].values,
            mean[col].values + stderr[col].values,
            alpha=alpha,
            **kwargs
        )
    return ax


def matplotlib_setup(size=24, use_tex=False):
    params = {
        'legend.fontsize': size * 0.75,
        'figure.figsize': (10, 5),
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size * 0.75,
        'ytick.labelsize': size * 0.75,
        'font.family': 'sans-serif',
        'axes.titlepad': 12.5,
        'text.usetex': use_tex
    }
    plt.rcParams.update(params)