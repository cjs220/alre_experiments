from experiments.mixtures_parameterized.analysis import analyse_mixtures_parameterized
from experiments.mixtures_parameterized.experiment import run_mixtures_parameterized
from util import ExperimentHandler


def run(config: str, n_experiments: int, n_jobs: int = 1, analysis: bool = True):
    handler = ExperimentHandler(
        __package__.split('.')[-1],
        config_name=config,
        run_func=run_mixtures_parameterized,
        analysis_func=analyse_mixtures_parameterized
    )
    handler.run_experiments(n_experiments=n_experiments, n_jobs=n_jobs)
    if analysis:
        handler.run_analysis()


if __name__ == '__main__':
    run(config='dev', n_experiments=1, n_jobs=1)
