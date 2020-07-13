import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import ExperimentHandler, experiment


@experiment
def dummy_experiment(a, b, c, logger=None):
    x = np.linspace(-1, 1, 1001)
    logging.info('Calculating y')
    y = x**2 + 0.2*np.random.rand(x.size)
    return dict(y=pd.DataFrame(index=x, data=y, columns=['y']))


def dummy_analysis(results, config):
    fig, ax = plt.subplots()
    y = pd.concat(results['y'], axis=1)
    y.columns = list(range(len(results['y'])))
    y.plot(ax=ax)
    return dict(y=fig)


def dummy_run():
    handler = ExperimentHandler(
        'dummy',
        'dummy',
        run_func=dummy_experiment,
        analysis_func=dummy_analysis
    )
    handler.run_experiments(n_experiments=10, n_jobs=1)
    handler.run_analysis()


if __name__ == '__main__':
    dummy_run()
