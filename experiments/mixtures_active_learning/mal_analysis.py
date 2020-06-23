from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.generic import NDFrame

from util import plot_line_graph_with_errors


def plot_mle_error(mle_err):
    fig, ax = plt.subplots()
    mean = mle_err.mean(axis=1, level=1)
    stderr = mle_err.sem(axis=1, level=1)
    plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
    return fig


def analyse_mixtures_active_learning(results: Dict[str, List[NDFrame]], config: Dict):
    mle = results['mle']

    mle_err = pd.concat(
        [df.subtract(df['Exact'], axis=0).drop('Exact', axis=1) for df in mle],
        axis=1,
        keys=range(len(mle))
    ).abs()

    mle_err_fig = plot_mle_error(mle_err)
    pass
