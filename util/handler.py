import logging
import os
import shutil
import sys
import time
from copy import copy
from pathlib import Path
from typing import Callable, List, Dict
from warnings import warn

import pandas as pd
import yaml
from joblib import Parallel, delayed
from pkg_resources import resource_filename

from util import save_results
from util.plotting import matplotlib_setup


class ExperimentHandler:

    def __init__(self,
                 experiment_name: str,
                 config_name: str,
                 run_func: Callable,
                 analysis_func: Callable,
                 logger_level=logging.INFO
                 ):
        self.config_name = config_name
        self.config_path = \
            resource_filename(
                'experiments',
                os.path.join(experiment_name, 'config', f'{config_name}.yml')
            )
        self.run_func = run_func
        self.analysis_func = analysis_func
        self.logger_level = logger_level

        with open(self.config_path, 'r') as infile:
            self.config = yaml.safe_load(infile)

        self.out_dir = os.path.join('runs', config_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.results = self._load_existing_results()

    def run_experiments(self, n_experiments: int, n_jobs: int = 1) -> None:
        random_seeds = self._get_new_random_seeds(n_experiments)
        run_dirs = self._init_run_dirs(random_seeds)
        loggers = [self._get_logger(run_dir) for run_dir in run_dirs]

        if n_jobs == 1:
            new_results = [
                self.run_func(
                    **self.config,
                    random_seed=seed,
                    logger=loggers[i]
                )
                for i, seed in enumerate(random_seeds)
            ]
        else:
            new_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.run_func)(
                    **self.config,
                    random_seed=seed,
                    logger=loggers[i]
                )
                for i, seed in enumerate(random_seeds)
            )

        self._save_results_csv(new_results, run_dirs)
        self._store_new_results(new_results, random_seeds)

    def run_analysis(self) -> None:
        matplotlib_setup(use_tex=False)
        results_list = list(self.results.values())
        figures = self.analysis_func(
            self._list_of_dict_to_dict_of_list(results_list),
            self.config
        )
        save_results(
            path=os.path.join('plots', self.config_name),
            figures=figures,
            exist_ok=True
        )

    def _load_existing_results(self) -> Dict[int, Dict]:
        existing_results = dict()  # Dict[random_seed] = Dict
        for dir_name in os.listdir(self.out_dir):
            dir_path = os.path.join(self.out_dir, dir_name)
            random_seed = self._check_config(dir_path)
            result = dict()
            csv_names = [fname for fname in os.listdir(dir_path)
                         if fname.endswith('.csv')]

            if not csv_names:
                delete_dir = input(f'Directory {dir_path} contains no csv files; '
                                   f'would you like to delete it? (y/n)') == 'y'
                if delete_dir:
                    shutil.rmtree(dir_path)
                    continue

            for csv_name in csv_names:
                key, _ = csv_name.split('.csv')
                result[key] = pd.read_csv(
                    os.path.join(dir_path, csv_name),
                    index_col=0
                )
            if result:
                assert random_seed not in existing_results, \
                    'Two experiments have been run with the same random seed'
                existing_results[random_seed] = result
        return existing_results

    @staticmethod
    def _save_results_csv(new_results: List[Dict],
                          run_dirs: List[str]):
        for run_dir, result in zip(run_dirs, new_results):
            for frame_name, frame in result.items():
                out_path = os.path.join(run_dir, frame_name + '.csv')
                frame.to_csv(out_path)

    def _store_new_results(self,
                           new_results: List[Dict],
                           random_seeds: List[int]):
        if new_results:
            new_results = dict(zip(random_seeds, new_results))
            self.results.update(new_results)

    def _check_config(self, dir_path: str) -> int:
        # check same config and extract the random seed
        config_path = os.path.join(dir_path, 'config.yml')
        with open(config_path, 'r') as infile:
            config = yaml.safe_load(infile)

        random_seed = config.pop('random_seed')

        if set(config.keys()) != set(self.config.keys()):
            warn(f'Config {config_path} has different keys to this config.')
        for item_name, item_val in config.items():
            if item_val != self.config.get(item_name, None):
                warn(f'Config {config_path} '
                     f'has different value for config item {item_name}.')

        return random_seed

    def _init_run_dir(self, random_seed: int) -> str:
        run_dir_name = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M%S')
        run_dir_path = os.path.join(self.out_dir, run_dir_name)
        os.mkdir(run_dir_path)
        new_config_path = os.path.join(run_dir_path, 'config.yml')
        new_config = copy(self.config)
        new_config['random_seed'] = random_seed
        with open(new_config_path, 'w+') as outfile:
            yaml.dump(new_config, outfile)

        return run_dir_path

    def _init_run_dirs(self, random_seeds: List[int]) -> List[str]:
        run_dirs = []
        for seed in random_seeds:
            run_dirs.append(self._init_run_dir(seed))
            time.sleep(1)
        return run_dirs

    def _get_logger(self, run_dir: str) -> logging.Logger:
        logger_name = os.path.join(run_dir, 'run.log')
        Path(logger_name).touch()

        logger = logging.getLogger(logger_name)
        logger.setLevel(self.logger_level)
        format_string = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        log_format = logging.Formatter(format_string)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(logger_name, mode='a')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def _list_of_dict_to_dict_of_list(list_of_dict: List[Dict]) -> Dict[str, List]:
        try:
            return {k: [d[k] for d in list_of_dict] for k in list_of_dict[0]}
        except IndexError:
            return {}

    @property
    def _trialed_seeds(self) -> List[int]:
        # random seeds for which the experiments have already been run
        return list(self.results.keys())

    def _get_new_random_seeds(self, n_experiments: int) -> List[int]:
        max_trialed_seed = max(self._trialed_seeds)
        random_seeds = list(range(max_trialed_seed + 1,
                                  max_trialed_seed + n_experiments + 1))
        return random_seeds
