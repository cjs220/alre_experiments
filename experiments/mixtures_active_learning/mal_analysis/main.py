from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame

idx = pd.IndexSlice

from experiments.mixtures_active_learning.mal_analysis.debug import \
    plot_predictions, _analyse_std
from experiments.mixtures_parameterized.mp_analysis import \
    TEST_STAT_ABBRV_STR, THETA_STR
from util.plotting import plot_line_graph_with_errors


def plot_mle_error(
        mle: List[pd.DataFrame],
        learner_names: List[str] = None,
        plot_median: bool = True,
        rolling: int = 50,
) -> Figure:
    learner_names = learner_names or slice(None)
    mle_err = pd.concat(
        [df.subtract(df['Exact'], axis=0).drop('Exact', axis=1) for df in mle],
        axis=1,
        keys=range(len(mle))
    ).abs()
    mle_err = mle_err.loc[:, (slice(None), learner_names)]

    nrows = 2 * (1 + plot_median)
    fig, axarr = plt.subplots(nrows=nrows, figsize=(10, nrows * 5))

    mean = mle_err.mean(axis=1, level=1)
    stderr = mle_err.sem(axis=1, level=1)
    plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=axarr[0])
    axarr[0].set(title='Mean absolute error on MLE estimate')

    rolling_mean = mean.rolling(rolling).mean()
    rolling_stderr = stderr.rolling(rolling).mean()
    plot_line_graph_with_errors(mean=rolling_mean, stderr=rolling_stderr, ax=axarr[1])
    axarr[1].set(title=f'Rolling ({rolling}) mean mean absolute error on MLE estimate')

    if plot_median:
        median = mle_err.mean(axis=1, level=1)
        ax = axarr[2]
        median.plot(ax=ax)
        ax.set(title='Median absolute error on MLE estimate')
        ax = axarr[3]
        median.rolling(rolling).mean().plot(ax=ax)
        ax.set(title=f'Rolling ({rolling}) mean median absolute error on MLE estimate')

    axarr[-1].set(xlabel='Active learning iteration')
    return fig


def plot_total_mse(
        test_stat: pd.DataFrame,
        test_stat_exact: pd.DataFrame,
        rolling: int = 50
) -> Figure:
    group_levels = ['Learner', 'Iteration']
    columns = test_stat.columns.get_level_values('Learner').unique()
    index = test_stat.columns.get_level_values('Iteration').unique()
    mean = pd.DataFrame(index=index, columns=columns)
    stderr = pd.DataFrame(index=index, columns=columns)
    gb = test_stat.groupby(axis=1, level=group_levels)

    for group in gb.groups.keys():
        pred_group = gb.get_group(group).droplevel(axis=1, level=group_levels)
        squared_error = (test_stat_exact - pred_group) ** 2
        mse_group = squared_error.mean(axis=0)
        mean_mse_group = mse_group.mean()
        stderr_group = mse_group.sem()
        learner, iteration = group
        mean.loc[iteration, learner] = mean_mse_group
        stderr.loc[iteration, learner] = stderr_group

    def _ensure_numeric(df):
        return df.astype(np.float64).set_index(df.index.astype(np.float64))

    mean, stderr = map(_ensure_numeric, [mean, stderr])

    fig, axarr = plt.subplots(2, figsize=(10, 10))
    plot_line_graph_with_errors(
        mean=mean,
        stderr=stderr,
        ax=axarr[0]
    )
    axarr[0].set(title='Total MSE', xlabel=None)

    rolling_mean = mean.rolling(rolling).mean()
    rolling_stderr = stderr.rolling(rolling).mean()
    plot_line_graph_with_errors(
        mean=rolling_mean,
        stderr=rolling_stderr,
        ax=axarr[1]
    )
    axarr[1].set(title=f'Rolling {rolling} mean total MSE',
                 xlabel='Active learning iteration')

    return fig


def plot_mse_vs_theta(
        test_stat: pd.DataFrame,
        test_stat_exact: pd.DataFrame,
        iterations: List[int]
):
    iterations = list(map(str, iterations))
    fig, axarr = plt.subplots(len(iterations), figsize=(10, len(iterations) * 5))
    for iteration, ax in zip(iterations, axarr):
        data = test_stat.loc[:, idx[:, :, iteration]].droplevel('Iteration', axis=1)
        gb = data.groupby(by='Learner', axis=1)
        squared_error = pd.concat([
            (df.droplevel('Learner', axis=1) - test_stat_exact) ** 2 for _, df in gb
        ], axis=1, keys=gb.groups.keys(), names=['Learner'])
        mean = squared_error.mean(axis=1, level='Learner')
        stderr = squared_error.sem(axis=1, level='Learner')
        plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
        ax.set(
            title=f'Iteration {iteration}',
            xlabel=None,
            ylabel='MSE',
            yscale='log'
        )
        ax.axvline(x=0.45, color='k', lw=2)

    axarr[-1].set(xlabel=THETA_STR)
    return fig


def plot_final_iteration_test_stat(
        test_stat: pd.DataFrame,
        test_stat_exact: pd.DataFrame
) -> Figure:
    alpha = 0.2
    learners = test_stat.columns.get_level_values('Learner').unique()
    fig, axarr = plt.subplots(
        nrows=len(learners),
        figsize=(10, 2.5 * len(learners))
    )
    ax_list = np.ravel(axarr).tolist()
    gb = test_stat.groupby(level='Learner', axis=1)
    for i, learner_name in enumerate(gb.groups.keys()):
        test_stat_group = (gb
                           .get_group(learner_name)
                           .droplevel(level='Learner', axis=1))
        ax = ax_list[i]
        test_stat_group.plot(ax=ax, alpha=alpha, color='r')
        test_stat_exact.plot(ax=ax, alpha=alpha, color='b')
        ax.legend().set_visible(False)
        ax.set_xlabel(None)
        ax.set_ylabel(TEST_STAT_ABBRV_STR, rotation=90, labelpad=5)
        ax.set(title=learner_name)
    axarr[-1].set(xlabel=THETA_STR)
    return fig


def plot_trialed_thetas_hist(trialed_thetas: List[pd.DataFrame]):
    pd.concat(trialed_thetas, axis=0).plot.hist(
        subplots=True,
        bins=80,
        figsize=(10, 20),
    )
    plt.xlabel(THETA_STR)
    return plt.gcf()


def plot_theta_convergence(
        trialed_thetas: List[pd.DataFrame],
        mle: List[pd.DataFrame],
        learner_names: List[str] = None,
        experiments: List[int] = None
):
    experiments = experiments or list(range(len(trialed_thetas)))

    def _concat(list_df):
        return pd.concat(
            list_df,
            axis=1,
            keys=range(len(trialed_thetas)),
            names=['Experiment', 'Learner']
        )

    trialed_df = _concat(trialed_thetas)
    mle_df = _concat(mle)
    mle_exact = (mle_df.
                 loc[:, (slice(None), 'Exact')]
                 .droplevel('Learner', axis=1))
    mle_df = mle_df.drop('Exact', axis=1, level='Learner')
    all_df = pd.concat(
        [trialed_df, mle_df],
        axis=1,
        keys=[r'Trialed $\theta$', 'MLE'],
        names=['Quantity']
    )

    if learner_names is None:
        learner_names = all_df.columns.get_level_values('Learner').unique()

    fig, axarr = plt.subplots(
        nrows=len(experiments),
        ncols=len(learner_names),
        figsize=(10 * len(learner_names), 5 * len(experiments))
    )

    for i, experiment in enumerate(experiments):
        for j, learner in enumerate(learner_names):
            ax = axarr[i, j]
            data = (all_df
                    .loc[:, (slice(None), experiment, learner)]
                    .droplevel(['Experiment', 'Learner'], axis=1))
            data.plot(ax=ax, marker='o')
            mle_exact.loc[:, experiment].plot(ax=ax, label='Exact MLE')
            ax.set(title=f'{learner} Experiment {experiment}')
            ax.legend()

    return fig


def analyse_mixtures_active_learning(
        results: Dict[str, List[NDFrame]],
        config: Dict
):
    # Loading and preprocessing
    mle = results['mle']
    nllr = _aggregrate_nllr_predictions(results['nllr'])
    std = _aggregrate_nllr_predictions(results['std'])
    nllr_exact = results['nllr_exact']
    trialed_thetas = results['trialed_thetas']
    test_stat = _calc_test_stat(nllr)
    test_stat_exact = pd.concat(map(_calc_test_stat, nllr_exact), axis=1)
    test_stat_exact.columns = range(len(nllr_exact))

    experiments = [0, 1, 2]
    iterations = np.linspace(0, 1000, 11).astype(np.int64)
    learner_names = ['Random', 'UCB_2.5', 'UCBM_2.5']

    mle_err_fig = plot_mle_error(
        mle=mle,
        learner_names=learner_names,
        plot_median=False
    )
    total_mse = plot_total_mse(
        test_stat=test_stat,
        test_stat_exact=test_stat_exact,
    )

    mse_vs_theta = plot_mse_vs_theta(
        test_stat=test_stat,
        test_stat_exact=test_stat_exact,
        iterations=[0, 100, 500, 1000]
    )

    # *****************************
    trialed_thetas_hist = plot_trialed_thetas_hist(
        trialed_thetas=trialed_thetas
    )
    convergence_plot = plot_theta_convergence(
        trialed_thetas=trialed_thetas,
        mle=mle,
        learner_names=learner_names,
        experiments=None
    )
    predictions_fig = plot_predictions(
        test_stat=test_stat,
        std=std,
        test_stat_exact=test_stat_exact,
        experiments=experiments,
        iterations=iterations,
        learner_names=None
    )
    std_fig = _analyse_std(
        test_stat=nllr,
        std=std,
        learner_names=['UCBM_2.5'],
        iterations=iterations,
        experiments=experiments,
    )

    # Collect figures
    figures = dict(
        mle_err=mle_err_fig,
        mse=total_mse,
        trialed_thetas=trialed_thetas_hist,
        convergence_plot=convergence_plot,
        mse_vs_theta=mse_vs_theta,
        predictions=predictions_fig,
        std=std_fig
    )

    return figures


def _aggregrate_nllr_predictions(
        predictions: List[pd.DataFrame],
):
    def _add_learner_level(df):
        learners = df['Learner'].unique()
        return pd.concat(
            [df[df['Learner'] == learner].drop('Learner', axis=1)
             for learner in learners],
            axis=1,
            keys=learners
        )

    aggregated = pd.concat(
        map(_add_learner_level, predictions),
        axis=1,
        keys=range(len(predictions)),
        names=('Experiment', 'Learner', 'Iteration')
    )
    aggregated.index.name = 'theta'
    return aggregated


def _calc_test_stat(nllr: pd.DataFrame) -> pd.DataFrame:
    return 2 * (nllr - nllr.min())
