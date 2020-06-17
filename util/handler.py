import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, List, Dict

import pandas as pd
import yaml
from joblib import Parallel, delayed
from pkg_resources import resource_filename

from util import matplotlib_setup, set_all_random_seeds, save_results


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
            resource_filename('experiments', os.path.join(experiment_name, 'config', f'{config_name}.yml'))
        self.run_func = run_func
        self.analysis_func = analysis_func
        self.logger_level = logger_level

        with open(self.config_path, 'r') as infile:
            self.config = yaml.safe_load(infile)

        self.out_dir = os.path.join('runs', config_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.results = self._load_existing_results()

    def run_experiments(self, n_experiments: int, n_jobs: int = 1) -> None:
        matplotlib_setup()
        set_all_random_seeds()

        run_dirs = []
        for i in range(n_experiments):
            run_dirs.append(self._init_run_dir())
            time.sleep(1)

        loggers = [self._get_logger(run_dir) for run_dir in run_dirs]

        if n_jobs == 1:
            new_results = [self.run_func(**self.config, logger=loggers[i]) for i in range(n_experiments)]
        else:
            new_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.run_func)(**self.config, logger=loggers[i]) for i in range(n_experiments)
            )

        for run_dir, result in zip(run_dirs, new_results):
            for frame_name, frame in result.items():
                out_path = os.path.join(run_dir, frame_name + '.csv')
                frame.to_csv(out_path)

        self.results += new_results

    def run_analysis(self) -> None:
        figures = self.analysis_func(self._list_of_dict_to_dict_of_list(self.results))
        save_results(path=os.path.join('plots', self.config_name), figures=figures, exist_ok=True)

    def _load_existing_results(self):
        existing_results = []
        for dir in os.listdir(self.out_dir):
            dir_path = os.path.join(self.out_dir, dir)
            result = dict()
            csv_names = [fname for fname in os.listdir(dir_path) if fname.endswith('.csv')]
            for csv_name in csv_names:
                key, _ = csv_name.split('.csv')
                result[key] = pd.read_csv(os.path.join(dir_path, csv_name), index_col=0)
            if result:
                existing_results.append(result)
        return existing_results

    def _init_run_dir(self):
        run_dir_name = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M%S')
        run_dir_path = os.path.join(self.out_dir, run_dir_name)
        os.mkdir(run_dir_path)
        shutil.copy(self.config_path, os.path.join(run_dir_path, 'config.yml'))
        return run_dir_path

    def _get_logger(self, run_dir):
        logger_name = os.path.join(run_dir, 'run.log')
        Path(logger_name).touch()

        logger = logging.getLogger(logger_name)
        logger.setLevel(self.logger_level)
        format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                         "%(lineno)d — %(message)s")
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
        return {k: [d[k] for d in list_of_dict] for k in list_of_dict[0]}
