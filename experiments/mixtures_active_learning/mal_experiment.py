import logging
import time
from logging import Logger
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

from active_learning_ratio_estimation.active_learning.active_learner import UpperConfidenceBoundLearner, \
    RandomActiveLearner
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
        ucb_kappa: float,
        logger: Logger = None
) -> Dict[str, NDFrame]:
    logger = logger or logging.getLogger(__name__)
    logger.info('Starting experiment')
    t0 = time.time()

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

    active_learners = dict(
        UCB=UpperConfidenceBoundLearner(
            simulator_func=triple_mixture,
            X_true=X_true,
            theta_true=theta_true,
            theta_0=theta_0,
            initial_idx=initial_idx,
            n_samples_per_theta=n_samples_per_theta,
            ratio_model=create_model(theta_0=theta_0, hyperparams=hyperparams),
            total_param_grid=param_grid,
            ucb_kappa=ucb_kappa
        ),
        Random=RandomActiveLearner(
            simulator_func=triple_mixture,
            X_true=X_true,
            theta_true=theta_true,
            theta_0=theta_0,
            initial_idx=initial_idx,
            n_samples_per_theta=n_samples_per_theta,
            ratio_model=create_model(theta_0=theta_0, hyperparams=hyperparams),
            total_param_grid=param_grid,
        ),
    )

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

    def _collect_nllr_info(learner_name, info):
        return pd.DataFrame(
            data=np.stack(getattr(active_learners[learner_name], info), axis=1),
            columns=[f'Iteration {i}' for i in range(n_iter + 1)],
            index=param_grid.array.squeeze()
        )

    random_nllr = _collect_nllr_info('Random', 'nllr_predictions')
    ucb_nllr = _collect_nllr_info('UCB', 'nllr_predictions')
    ucb_std = _collect_nllr_info('UCB', 'nllr_std')
    random_nllr['Exact'] = nllr_exact.squeeze()
    ucb_nllr['Exact'] = nllr_exact.squeeze()

    logger.info(f'Finished experiment; total time {int(time.time() - t0):.3E} s')
    return dict(
        mle=mle,
        trialed_thetas=trialed_thetas,
        random_nllr=random_nllr,
        ucb_nllr=ucb_nllr,
        ucb_std=ucb_std
    )
