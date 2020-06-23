import logging
import time
from logging import Logger
from typing import Dict, List

from active_learning_ratio_estimation.active_learning.active_learner import UpperConfidenceBoundLearner, \
    RandomActiveLearner
from active_learning_ratio_estimation.dataset import ParamGrid
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel
from util.distributions import triple_mixture


def create_model(theta_0: float, hyperparams: Dict):
    clf = FlipoutClassifier(**hyperparams)
    return SinglyParameterizedRatioModel(theta_0=theta_0, clf=clf)


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
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Starting experiment')
    t0 = time.time()

    X_true = triple_mixture(theta_true).sample(n_true).numpy()

    param_grid = ParamGrid(bounds=[theta_bounds], num=n_theta)

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

    for name, active_learner in active_learners.items():
        active_learner.fit(n_iter=n_iter)

    pass
