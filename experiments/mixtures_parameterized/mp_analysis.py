import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame

from util.plotting import plot_line_graph_with_errors

TEST_STAT_STR = r'$-2 \, \log \frac{L(\theta)}{L(\theta_{MLE})}$'
THETA_STR = r'$\theta$'
TEST_STAT_ABBRV_STR = r'$q(\theta)$'


def plot_mse_vs_theta(mse: pd.DataFrame, trimmed_mse: pd.DataFrame) -> Figure:
    fig, axarr = plt.subplots(2, sharex=True)

    def _plot_mse(df, ax):
        mean = df.mean(axis=0, level=1)
        stderr = df.sem(axis=0, level=1)
        plot_line_graph_with_errors(mean, stderr, ax=ax)

    for ax, df, title in zip(axarr, [mse, trimmed_mse], ['MSE', 'Trimmed MSE']):
        _plot_mse(df, ax)
        ax.set_title(title)

    axarr[1].set_xlabel(THETA_STR)
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


def plot_average_test_stat(test_stat: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    mean = test_stat.mean(axis=0, level=1)
    stderr = test_stat.sem(axis=0, level=1)
    ax = plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
    ax.set(xlabel=THETA_STR, title=TEST_STAT_STR)
    return fig


def plot_test_stat(test_stat: pd.DataFrame) -> Figure:
    fig, axarr = plt.subplots(test_stat.shape[1] - 1, sharex=True, figsize=(17, 10))
    non_exact_cols = filter(lambda x: x != 'Exact', test_stat.columns)
    alpha = 0.1
    for ax, col in zip(axarr, non_exact_cols):
        test_stat[col].unstack(0).plot(ax=ax, alpha=alpha, color='r', label=None)
        test_stat['Exact'].unstack(0).plot(ax=ax, alpha=alpha, color='b', label=None)
        ax.set(title=col, ylim=(0, 600), xlim=(0.1, 0.9))
        ax.legend().set_visible(False)
        ax.set_ylabel(TEST_STAT_ABBRV_STR, rotation=90, labelpad=5)
    axarr[-1].set(xlabel=THETA_STR)
    return fig


def plot_test_stat_mse(test_stat: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    squared_error = (test_stat.subtract(test_stat['Exact'], axis=0)**2).drop('Exact', axis=1)
    mean_squared_error = squared_error.mean(axis=0, level=1)
    stderr_squared_error = squared_error.sem(axis=0, level=1)
    ax = plot_line_graph_with_errors(mean=mean_squared_error, stderr=stderr_squared_error, ax=ax)
    ax.set(yscale='log', xlabel=THETA_STR, title=f'Squared error on {TEST_STAT_ABBRV_STR}={TEST_STAT_STR}')
    return fig


def plot_mle_distributions(
        mle: pd.DataFrame,
        theta_true: float,
        theta_bounds_train: Sequence[float]
) -> Figure:
    fig, axarr = plt.subplots(mle.shape[1] - 1, sharex=True, figsize=(10, 15))
    non_exact_cols = filter(lambda x: x != 'Exact', mle.columns)
    for ax, col in zip(axarr, non_exact_cols):
        for model in ('Exact', col):
            mle[model].plot.hist(ax=ax, alpha=0.3, density=True)
        ax.set(title=col, xlim=theta_bounds_train)
        ax.legend()
        ax.set_prop_cycle(None)
        try:
            for model in ('Exact', col):
                mle[model].plot.kde(ax=ax, label=None)
        except np.linalg.LinAlgError:
            pass
        ax.axvline(
            x=theta_true,
            color='k',
            linestyle='--'
        )
        ax.set_ylabel(None)
    axarr[-1].set(xlabel=THETA_STR)
    return fig


def plot_bias_variance(mle: pd.DataFrame, theta_true: float) -> Figure:
    bias = (mle - theta_true).mean().abs()
    variance = mle.var()
    quantities = {'Absolute Bias': bias, 'Variance': variance}
    fig, axarr = plt.subplots(2, sharex=True)
    for (quantity_name, quantity), ax in zip(quantities.items(), axarr):
        quantity.plot.bar(alpha=0.5, ax=ax, rot=0)
        ax.set(title=quantity_name)
    return fig


def analyse_mixtures_parameterized(results: Dict[str, List[NDFrame]], config: Dict):
    theta_true = config['theta_true']
    theta_bounds_train = config['theta_bounds_train']

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
    test_stat_fig = plot_test_stat(test_stat)
    average_test_stat_fig = plot_average_test_stat(test_stat)
    mse_on_test_stat_fig = plot_test_stat_mse(test_stat)

    mle = pd.concat(mle, axis=1).T
    mle_fig = plot_mle_distributions(mle=mle, theta_true=theta_true, theta_bounds_train=theta_bounds_train)
    bias_variance_fig = plot_bias_variance(mle=mle, theta_true=theta_true)

    figures = dict(
        mse_vs_theta=mse_vs_theta_fig,
        total_mse=total_mse_fig,
        average_test_stat=average_test_stat_fig,
        test_stat=test_stat_fig,
        mse_on_test_stat=mse_on_test_stat_fig,
        mle=mle_fig,
        bias_variance=bias_variance_fig
    )
    return figures
