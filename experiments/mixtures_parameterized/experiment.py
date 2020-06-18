import logging
import time
from logging import Logger
from typing import Sequence, Dict

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from active_learning_ratio_estimation.model.ratio_model import calibrated_param_scan, param_scan, exact_param_scan
from active_learning_ratio_estimation.dataset import SinglyParameterizedRatioDataset, ParamGrid, ParamIterator
from active_learning_ratio_estimation.model import DenseClassifier, SinglyParameterizedRatioModel, FlipoutClassifier

from util.distributions import triple_mixture


def create_models(
        theta_0: float,
        hyperparams: Dict,
) -> Dict[str, SinglyParameterizedRatioModel]:
    # regular, uncalibrated model
    regular_estimator = DenseClassifier(activation='tanh', **hyperparams)
    regular_uncalibrated = SinglyParameterizedRatioModel(theta_0=theta_0, clf=regular_estimator)

    # bayesian, uncalibrated model
    bayesian_estimator = FlipoutClassifier(activation='relu', **hyperparams)
    bayesian_uncalibrated = SinglyParameterizedRatioModel(theta_0=theta_0, clf=bayesian_estimator)

    models = {
        'Regular': regular_uncalibrated,
        'Bayesian': bayesian_uncalibrated,
    }
    return models


def fit_and_predict_models(
        models: Dict[str, SinglyParameterizedRatioModel],
        train_dataset: SinglyParameterizedRatioDataset,
        test_dataset: SinglyParameterizedRatioDataset,
) -> pd.DataFrame:
    predictions = pd.DataFrame(dict(
        theta=test_dataset.theta_1s.squeeze(),
        x=test_dataset.x.squeeze(),
        Exact=(test_dataset.log_prob_1 - test_dataset.log_prob_0).squeeze()
    ))
    for model_name, model in models.items():
        model.fit(train_dataset.x, train_dataset.theta_1s, train_dataset.y)
        logr = model.predict(test_dataset.x, test_dataset.theta_1s, log=True)
        predictions[model_name] = logr
    return predictions


def calibrated_predict(
        model: SinglyParameterizedRatioModel,
        test_dataset: SinglyParameterizedRatioDataset,
        n_calibration: int,
        calibration_params: Dict,
) -> np.ndarray:
    predictions = np.array([])
    for theta in np.unique(test_dataset.theta_1s, axis=0):
        dataset_slice = test_dataset[np.all(test_dataset.theta_1s == theta, axis=1)]
        new_predictions = model.calibrated_predict(
            X=dataset_slice.x,
            theta=theta,
            n_samples_per_theta=n_calibration,
            simulator_func=triple_mixture,
            calibration_params=calibration_params,
            log=True,
        )
        predictions = np.append(predictions, new_predictions)
    return predictions


def run_param_scan(
        model: SinglyParameterizedRatioModel,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        n_calibration: int = None,
        calibration_params: Dict = None
):
    if calibration_params is not None:
        return calibrated_param_scan(
            model=model,
            X_true=X_true,
            param_grid=param_grid,
            simulator_func=triple_mixture,
            n_samples_per_theta=n_calibration,
            calibration_params=calibration_params
        )
    else:
        return param_scan(
            model=model,
            X_true=X_true,
            param_grid=param_grid,
        )


def run_mixtures_parameterized(
        theta_0: float,
        theta_bounds_train: Sequence[float],
        n_samples_per_theta_train: int,
        n_thetas_train: int,
        theta_test_values: np.array,
        n_samples_per_theta_test: int,
        hyperparams: Dict,
        calibration_params: Dict,
        n_calibration: int,
        theta_true: float,
        n_true: int,
        logger: Logger = None
) -> Sequence[pd.DataFrame]:
    logger = logger or logging.getLogger(__name__)
    logger.info('Starting experiment')
    t0 = time.time()

    logger.info('Building training dataset')
    train_grid = ParamGrid(bounds=[theta_bounds_train], num=n_thetas_train)
    train_ds = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        n_samples_per_theta=n_samples_per_theta_train,
        theta_1_iterator=train_grid,
        include_log_probs=False
    )

    logger.info('Building test dataset')
    theta_test_values = np.array(theta_test_values).reshape(-1, 1)
    test_iterator = ParamIterator([theta for theta in theta_test_values])
    test_ds = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        n_samples_per_theta=n_samples_per_theta_test,
        theta_1_iterator=test_iterator,
        include_log_probs=True
    )
    test_ds = test_ds[test_ds.y == 1]

    logger.info('Fitting and predicting models')
    # TODO fit and predict as separate functions so can log
    models = create_models(theta_0=theta_0, hyperparams=hyperparams)
    predictions = fit_and_predict_models(
        models=models,
        train_dataset=train_ds,
        test_dataset=test_ds
    )

    logger.info('Performing calibrated predict')
    predictions['Calibrated'] = calibrated_predict(
        model=models['Regular'],
        test_dataset=test_ds,
        n_calibration=n_calibration,
        calibration_params=calibration_params
    )

    logger.info('Calculating expected MSE')
    predictions = predictions.sort_values(by='theta').set_index(['theta', 'x'])
    squared_errors = ((predictions.subtract(predictions['Exact'], axis=0)) ** 2).drop('Exact', axis=1)
    mse = squared_errors.mean(level=0)
    trimmed_mse = pd.concat(
        [squared_errors[col].groupby('theta').apply(trim_mean, 0.05) for col in squared_errors.columns],
        axis=1
    )

    logger.info('Running regular param scans')
    X_true = triple_mixture(theta_true).sample(n_true).numpy()
    param_scan_results = {
        model_name: run_param_scan(model=model, X_true=X_true, param_grid=train_grid)
        for model_name, model in models.items()
    }

    logger.info('Running calibrated param scan')
    param_scan_results['Calibrated'] = run_param_scan(
        model=models['Regular'],
        X_true=X_true,
        param_grid=train_grid,
        n_calibration=n_calibration,
        calibration_params=calibration_params
    )
    nllr, mle = zip(*param_scan_results.values())
    nllr = pd.DataFrame(dict(zip(param_scan_results.keys(), nllr)), index=train_grid.array.squeeze())
    mle = pd.Series(dict(zip(param_scan_results.keys(), np.array(mle).squeeze())))

    logger.info('Running exact param scan')
    nllr_exact, mle_exact = exact_param_scan(
        simulator_func=triple_mixture,
        X_true=X_true,
        param_grid=train_grid,
        theta_0=theta_0
    )
    nllr['Exact'] = nllr_exact
    mle['Exact'] = mle_exact.squeeze()
    test_stat = 2 * (nllr - nllr.min())

    logger.info(f'Finished experiment; total time {int(time.time() - t0):.3E} s')
    return dict(mse=mse, trimmed_mse=trimmed_mse, test_stat=test_stat, mle=mle)
