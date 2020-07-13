import logging
from logging import Logger
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

from active_learning_ratio_estimation.active_learning.active_learner import UpperConfidenceBoundLearner, \
    RandomActiveLearner, ModifiedUCBLearner
from active_learning_ratio_estimation.dataset import ParamGrid
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel
from active_learning_ratio_estimation.model.ratio_model import exact_param_scan

from util import experiment
from util.distributions import triple_mixture


def create_model(theta_0: float, hyperparams: Dict):
    clf = FlipoutClassifier(**hyperparams)
    return SinglyParameterizedRatioModel(theta_0=theta_0, clf=clf)


@experiment
def run_mixtures_active_learning(
        theta_0: float,
        theta_true: float,
        theta_bounds: List[float],
        n_theta: int,
        initial_idx: List[int],
        hyperparams: Dict,
        n_true: int,
        n_samples_per_theta: int,
        n_iter: int,
        ucb_kappas: List[float],
        ucbm_kappas: List[float],
        logger: Logger = None
) -> Dict[str, NDFrame]:
    logger = logger or logging.getLogger(__name__)

    logger.info('Simulating X_true and performing exact param scan')
    param_grid = ParamGrid(bounds=[theta_bounds], num=n_theta)
    X_true = triple_mixture(theta_true).sample(n_true).numpy()
    nllr_exact, mle_exact = exact_param_scan(
        simulator_func=triple_mixture,
        X_true=X_true,
        param_grid=param_grid,
        theta_0=theta_0,
        to_meshgrid_shape=False
    )

    logger.info('Building active learners')
    learner_kwargs = dict(
        simulator_func=triple_mixture,
        X_true=X_true,
        theta_true=theta_true,
        theta_0=theta_0,
        initial_idx=initial_idx,
        n_samples_per_theta=n_samples_per_theta,
        ratio_model=create_model(theta_0=theta_0, hyperparams=hyperparams),
        total_param_grid=param_grid,
    )
    active_learners = dict(Random=RandomActiveLearner(**learner_kwargs))

    for ucb_kappa in ucb_kappas:
        active_learners[f'UCB_{ucb_kappa}'] = \
            UpperConfidenceBoundLearner(kappa=ucb_kappa, **learner_kwargs)

    for ucbm_kappa in ucbm_kappas:
        active_learners[f'UCBM_{ucbm_kappa}'] = \
            ModifiedUCBLearner(kappa=ucbm_kappa, **learner_kwargs)

    logger.info('Fitting ActiveLearners.')
    for name, active_learner in active_learners.items():
        logger.info(f'Fitting {name} ActiveLearner.')
        active_learner.fit(n_iter=n_iter)

    logger.info('Finished fitting, collecting results.')

    mle = pd.DataFrame(
        {learner_name: map(float, learner.mle_predictions)
         for learner_name, learner in active_learners.items()}
    )
    mle['Exact'] = float(mle_exact)

    trialed_thetas = pd.DataFrame(
        {learner_name: map(float, learner.trialed_thetas)
         for learner_name, learner in active_learners.items()}
    )

    all_thetas = np.around(param_grid.array.squeeze(), 6)  # TODO

    def _collect_predictions(attr_name):
        columns = list(range(n_iter + 1))
        default = [np.full((len(all_thetas),), np.nan)
                   for _ in range(len(columns))]
        dfs = [
            pd.DataFrame(
                data=np.stack(getattr(learner, attr_name, default), axis=1),
                index=all_thetas,
                columns=columns
            )
            for learner in active_learners.values()
        ]
        concat = pd.concat(
            dfs,
            axis=0,
            keys=active_learners.keys(),
            names=['Learner', 'theta']
        )
        concat = concat.reset_index().set_index('theta', drop=True)
        return concat

    nllr = _collect_predictions('nllr_predictions')
    std = _collect_predictions('nllr_std')
    nllr_exact = pd.DataFrame(data=nllr_exact.squeeze(), columns=['Exact'], index=all_thetas)

    return dict(
        mle=mle,
        trialed_thetas=trialed_thetas,
        nllr=nllr,
        std=std,
        nllr_exact=nllr_exact
    )
