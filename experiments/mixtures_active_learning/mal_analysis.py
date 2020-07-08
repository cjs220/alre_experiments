from typing import Dict, List, Sequence, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame

from experiments.mixtures_parameterized.mp_analysis import TEST_STAT_ABBRV_STR, THETA_STR
from util.plotting import plot_line_graph_with_errors


def plot_mle_error(mle_err) -> Figure:
    fig, ax = plt.subplots()
    mean = mle_err.mean(axis=1, level=1)
    stderr = mle_err.sem(axis=1, level=1)
    plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
    ax.set(title='MAE on MLE estimate')
    return fig


def plot_total_mse(
        test_stat: pd.DataFrame,
        test_stat_exact: pd.DataFrame
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

    fig, ax = plt.subplots()
    plot_line_graph_with_errors(
        mean=mean.astype(np.float64),
        stderr=stderr.astype(np.float64),
        ax=ax
    )
    ax.set(title='Total MSE', xlabel='Active learning iteration')

    return fig


def plot_final_iteration_test_stat(
        test_stat: pd.DataFrame,
        test_stat_exact: pd.DataFrame
) -> Figure:
    alpha = 0.2
    learners = test_stat.columns.get_level_values('Learner').unique()
    fig, axarr = plt.subplots(nrows=len(learners), figsize=(10, 2.5 * len(learners)))
    ax_list = np.ravel(axarr).tolist()
    gb = test_stat.groupby(level='Learner', axis=1)
    for i, learner_name in enumerate(gb.groups.keys()):
        test_stat_group = gb.get_group(learner_name).droplevel(level='Learner', axis=1)
        ax = ax_list[i]
        test_stat_group.plot(ax=ax, alpha=alpha, color='r')
        test_stat_exact.plot(ax=ax, alpha=alpha, color='b')
        ax.legend().set_visible(False)
        ax.set_xlabel(None)
        ax.set_ylabel(TEST_STAT_ABBRV_STR, rotation=90, labelpad=5)
        ax.set(title=learner_name)
    axarr[-1].set(xlabel=THETA_STR)
    return fig


def analyse_mixtures_active_learning(
        results: Dict[str, List[NDFrame]],
        config: Dict
):
    mle = results['mle']
    nllr = _aggregrate_nllr_predictions(results['nllr'])
    std = _aggregrate_nllr_predictions(results['std'])
    nllr_exact = results['nllr_exact']

    mle_err = pd.concat(
        [df.subtract(df['Exact'], axis=0).drop('Exact', axis=1) for df in mle],
        axis=1,
        keys=range(len(mle))
    ).abs()
    mle_err_fig = plot_mle_error(mle_err)

    test_stat = _calc_test_stat(nllr)
    test_stat_exact = pd.concat(map(_calc_test_stat, nllr_exact), axis=1)
    test_stat_exact.columns = range(len(nllr_exact))

    mse_fig = plot_total_mse(test_stat=test_stat, test_stat_exact=test_stat_exact)
    test_stat_fig = plot_final_iteration_test_stat(
        test_stat=test_stat,
        test_stat_exact=test_stat_exact
    )

    experiments = [0, 1, 2]
    iterations = None

    debug_fig = _plot_debug_graph(
        test_stat=test_stat,
        std=std,
        test_stat_exact=test_stat_exact,
        experiments=experiments,
        iterations=iterations,
    )
    ucb_debug_fig = _plot_ucb_debug_graph(
        test_stat=test_stat,
        std=std,
        test_stat_exact=test_stat_exact,
        learner_name='UCB_0',
        kappas=[0, 15, -15],
        ns=[2, 3],
        iterations=iterations,
        experiments=experiments,
    )

    figures = dict(
        mle_err=mle_err_fig,
        mse=mse_fig,
        test_stat=test_stat_fig
    )

    return figures


def _plot_debug_graph(
        test_stat: pd.DataFrame,
        std: pd.DataFrame,
        test_stat_exact: pd.DataFrame,
        experiments: List[int] = None,
        iterations: List[int] = None,
        learner_names: List[str] = None
) -> Figure:
    learner_names = learner_names \
                    or test_stat.columns.get_level_values('Learner').unique()

    def _plotting_func(
            ax,
            experiment,
            iteration,
            test_stat,
            std,
            test_stat_exact,
            experiment_filter,
            iteration_filter
    ):
        iteration_filter = \
            test_stat.columns.get_level_values('Iteration') == str(iteration)

        learner_filter = \
            np.in1d(test_stat.columns.get_level_values('Learner'), learner_names)

        mask = iteration_filter & learner_filter & experiment_filter
        mean = (test_stat
                .loc[:, mask]
                .droplevel(['Experiment', 'Iteration'], axis=1)
                )
        stderr = (std
                  .loc[:, mask]
                  .fillna(0)
                  .droplevel(['Experiment', 'Iteration'], axis=1)
                  )
        plot_line_graph_with_errors(
            mean=mean,
            stderr=stderr,
            ax=ax
        )
        (test_stat_exact
         .iloc[:, experiment]
         .plot(color='k', lw=2, label='Exact', ax=ax))
        ax.set(
            title=f'Experiment {experiment} Iteration {iteration}',
            xlabel=None,
            ylabel=TEST_STAT_ABBRV_STR
        )
        ax.legend()

    fig, _ = _plot_per_experiment_and_iter(
        test_stat=test_stat,
        std=std,
        test_stat_exact=test_stat_exact,
        iterations=iterations,
        experiments=experiments,
        plotting_func=_plotting_func
    )

    return fig


def _plot_ucb_debug_graph(
        test_stat: pd.DataFrame,
        std: pd.DataFrame,
        test_stat_exact: pd.DataFrame,
        learner_name: str,
        kappas: List[float],
        ns: List[int],
        iterations: List[int] = None,
        experiments: List[int] = None,
) -> Figure:

    def _plotting_func(
            ax,
            experiment,
            iteration,
            test_stat,
            std,
            test_stat_exact,
            experiment_filter,
            iteration_filter
    ):
        learner_filter = \
            test_stat.columns.get_level_values('Learner') == learner_name
        mask = iteration_filter & learner_filter & experiment_filter
        mu = test_stat.loc[:, mask]
        sigma = std.loc[:, mask]

        kappa_df = pd.DataFrame({
            kappa: (-mu + kappa * sigma).values.squeeze()
            for kappa in kappas
        },
            index=mu.index
        ).rename(lambda x: rf'$\kappa={x}$', axis=1)

        n_df = pd.DataFrame({
            n: (-mu + sigma ** n).values.squeeze()
            for n in ns
        },
            index=mu.index
        ).rename(lambda x: rf'$n={x}$', axis=1)

        kappa_df.plot(ax=ax, ls='--')
        n_df.plot(ax=ax)
        (-1 * test_stat_exact.iloc[:, experiment]).plot(ax=ax, lw=2, label='Exact')
        _plot_maxima(kappa_df, ax)
        _plot_maxima(n_df, ax)

        ax.legend(ncol=2)
        ax.set(
            xlabel=None,
            title=f'Experiment {experiment} Iteration {iteration}'
        )
    fig, _ = _plot_per_experiment_and_iter(
        test_stat=test_stat,
        std=std,
        test_stat_exact=test_stat_exact,
        iterations=iterations,
        experiments=experiments,
        plotting_func=_plotting_func
    )
    return fig


def _plot_per_experiment_and_iter(
        test_stat: pd.DataFrame,
        std: pd.DataFrame,
        test_stat_exact: pd.DataFrame,
        iterations: List[int],
        experiments: List[int],
        plotting_func: Callable
):
    iterations = iterations or \
                 test_stat.columns.get_level_values('Iteration').unique()
    experiments = experiments or \
                  test_stat.columns.get_level_values('Experiment').unique()

    fig, axarr = plt.subplots(
        nrows=len(iterations),
        ncols=len(experiments),
        figsize=(10 * len(experiments), 5 * len(iterations))
    )
    for i, experiment in enumerate(experiments):
        for j, iteration in enumerate(iterations):
            ax = axarr[j, i]
            experiment_filter = \
                test_stat.columns.get_level_values('Experiment') == experiment
            iteration_filter = \
                test_stat.columns.get_level_values('Iteration') == str(iteration)
            plotting_func(
                ax=ax,
                experiment=experiment,
                iteration=iteration,
                test_stat=test_stat,
                std=std,
                test_stat_exact=test_stat_exact,
                experiment_filter=experiment_filter,
                iteration_filter=iteration_filter
            )
    return fig, axarr


def _plot_maxima(df, ax, **kwargs):
    for x, y in zip(df.idxmax(), df.max()):
        ax.plot([x], [y], **kwargs)


def _aggregrate_nllr_predictions(
        predictions: List[pd.DataFrame],
):
    def _add_learner_level(df):
        learners = df['Learner'].unique()
        return pd.concat(
            [df[df['Learner'] == learner].drop('Learner', axis=1) for learner in learners],
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
