from typing import List, Callable, Union, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from experiments.mixtures_parameterized.mp_analysis import TEST_STAT_ABBRV_STR
from util.plotting import plot_line_graph_with_errors


class MultiExperimentPlotter:

    def __init__(self,
                 plotting_func: Callable,
                 columns: pd.Index,
                 iterations: Union[List[int], None],
                 experiments: Union[List[int], None],
                 learner_names: Union[List[str], None],
                 ):
        self.plotting_func = plotting_func
        self.columns = columns
        self.iterations = (iterations or
                           self._get_default_col_vals(columns=columns, level='Iteration'))
        self.experiments = (experiments or
                            self._get_default_col_vals(columns=columns, level='Iteration'))
        self.learner_names = (learner_names or
                              self._get_default_col_vals(columns=columns, level='Learner'))
        self.learner_filter = np.in1d(columns.get_level_values('Learner'), learner_names)

    def __call__(self):
        fig, axarr = self._init_figure()
        for i, experiment in enumerate(self.experiments):
            for j, iteration in enumerate(self.iterations):
                ax = axarr[j, i]
                experiment_filter = self._filter_on_experiment(experiment)
                iteration_filter = self._filter_on_iteration(iteration)
                mask = experiment_filter & iteration_filter & self.learner_filter
                self.plotting_func(ax=ax, mask=mask)
                ax.set(
                    xlabel=None,
                    title=f'Experiment {experiment} Iteration {iteration}'
                )
        return fig, axarr

    def _init_figure(self):
        fig, axarr = plt.subplots(
            nrows=len(self.iterations),
            ncols=len(self.experiments),
            figsize=(10 * len(self.experiments), 5 * len(self.iterations))
        )
        return fig, axarr

    def _filter_on_experiment(self, experiment):
        return self._filter_on_value(columns=self.columns,
                                     level='Experiment',
                                     value=experiment)

    def _filter_on_iteration(self, iteration):
        return self._filter_on_value(columns=self.columns,
                                     level='Iteration',
                                     value=str(iteration))

    @staticmethod
    def _get_default_col_vals(columns: pd.Index, level: str):
        return columns.get_level_values(level=level).unique()

    @staticmethod
    def _filter_on_value(columns: pd.Index, level: str, value: Any):
        return columns.get_level_values(level=level) == value


def plot_debug_graph(
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


def plot_ucb_debug_graph(
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


def _analyse_std(
        nllr: pd.DataFrame,
        std: pd.DataFrame,
        learner_names: List[str] = None,
        experiments: List[int] = None,
        iterations: List[int] = None,
):
    def _plotting_func(ax, mask):
        mu = nllr.loc[:, mask]
        sigma = std.loc[:, mask]
        fractional_sigma = ((sigma / mu.abs())
                            .droplevel(['Iteration', 'Experiment'], axis=1)
                            )
        fractional_sigma.plot(ax=ax)

    plotter = MultiExperimentPlotter(
        plotting_func=_plotting_func,
        columns=nllr.columns,
        iterations=iterations,
        experiments=experiments,
        learner_names=learner_names
    )

    fig, _ = plotter()
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
