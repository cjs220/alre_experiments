import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame

from util import plot_line_graph_with_errors


def plot_mse_vs_theta(mse: pd.DataFrame, trimmed_mse: pd.DataFrame) -> Figure:
    fig, axarr = plt.subplots(2, sharex=True)

    def _plot_mse(df, ax):
        mean = df.mean(axis=0, level=1)
        stderr = df.sem(axis=0, level=1)
        plot_line_graph_with_errors(mean, stderr, ax=ax)

    for ax, df, title in zip(axarr, [mse, trimmed_mse], ['MSE', 'Trimmed MSE']):
        _plot_mse(df, ax)
        ax.set_title(title)

    axarr[1].set_xlabel(r'$\theta$')
    return fig


def plot_total_mse(mse: pd.DataFrame, trimmed_mse: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    average_errors = {'MSE': mse.mean(level=0), 'Trimmed MSE': trimmed_mse.mean(level=0)}

    mean_average_errors = pd.concat(
        [average_err.mean(axis=0).rename(err_name) for err_name, average_err in average_errors.items()],
        axis=1
    ).T

    stderr_average_errors = pd.concat(
        [average_err.sem(axis=0).rename(err_name) for err_name, average_err in average_errors.items()],
        axis=1
    ).T

    if (mean_average_errors == np.inf).any().any():
        logging.warning('Infinite values in expected MSE; replacing with np.nan')
        mean_average_errors = mean_average_errors.replace(np.inf, np.nan)
        stderr_average_errors = stderr_average_errors.replace(np.inf, np.nan)

    mean_average_errors.plot.bar(
        ax=ax,
        yerr=stderr_average_errors,
        alpha=0.5,
        capsize=10,
        rot=0,
    )
    return fig


def plot_test_stat(test_stat: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    mean = test_stat.mean(axis=0, level=1)
    stderr = test_stat.sem(axis=0, level=1)
    ax = plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
    ax.set(xlabel=r'$\theta$', title=r'$-2 \, \log \frac{L(\theta)}{L(\theta_{MLE})}$')
    return fig


def plot_mle_distributions(mle: pd.DataFrame, theta_true: float) -> Figure:
    mle = mle.astype(np.float32)
    fig, axarr = plt.subplots(mle.shape[1] - 1, sharex=True)
    non_exact_cols = filter(lambda x: x != 'Exact', mle.columns)
    for ax, col in zip(axarr, non_exact_cols):
        for model in ('Exact', col):
            mle[model].plot.hist(ax=ax, alpha=0.3, density=True)
        # TODO
        # ax.set_prop_cycle(None)
        # for model in ('Exact', col):
        #     mle[model].plot.kde(ax=ax, label=None)
        ax.axvline(x=theta_true, label='True', color='k', linestyle='--')
        ax.legend()
        ax.set_ylabel(None)
    return fig


def analyse_mixtures_parameterized(results: Dict[str, List[NDFrame]]):
    mse = results['mse']
    trimmed_mse = results['trimmed_mse']
    test_stat = results['test_stat']
    mle = results['mle']
    n_experiments, = set(map(len, results.values()))
    mse = pd.concat(mse, axis=0, keys=range(n_experiments))
    trimmed_mse = pd.concat(trimmed_mse, axis=0, keys=range(n_experiments))
    mse_vs_theta_fig = plot_mse_vs_theta(mse, trimmed_mse)
    total_mse_fig = plot_total_mse(mse, trimmed_mse)

    test_stat = pd.concat(test_stat, axis=0, keys=range(n_experiments))
    exclusion_fig = plot_test_stat(test_stat)

    mle = pd.concat(mle, axis=1).T
    mle_fig = plot_mle_distributions(mle=mle, theta_true=0.05)  # TODO: theta_true hard coded here
    figures = dict(mse_vs_theta=mse_vs_theta_fig, total_mse=total_mse_fig, exclusion=exclusion_fig, mle=mle_fig)
    return figures
