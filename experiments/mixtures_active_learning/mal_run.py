from argparse import ArgumentParser

from experiments.mixtures_active_learning.mal_experiment import run_mixtures_active_learning
from util import ExperimentHandler


parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True, help='Which config to use', type=str)
parser.add_argument('-e', '--experiments', required=True, help='Number of experiments', type=int)
parser.add_argument('-j', '--jobs', required=True, help='Number of jobs for multiprocessing', type=int)


def run(config: str, n_experiments: int, n_jobs: int = 1):
    handler = ExperimentHandler(
        'mixtures_active_learning',
        config_name=config,
        run_func=run_mixtures_active_learning,
        analysis_func=None
    )
    handler.run_experiments(n_experiments=n_experiments, n_jobs=n_jobs)
    handler.run_analysis()


if __name__ == '__main__':
    cmd_args = parser.parse_args()
    run(config=cmd_args.config, n_experiments=cmd_args.experiments, n_jobs=cmd_args.jobs)
