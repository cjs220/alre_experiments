import os

from pkg_resources import resource_filename

from active_learning_ratio_estimation.simulator import SimulationDatabase

DATA_DIR = resource_filename('experiments',
                             'eft_parameterized/data')


class EFTSimulationDatabase(SimulationDatabase):

    def __init__(self, theta: float):
        super().__init__(
            theta=theta,
            filepath=os.path.join(DATA_DIR, f'{theta}.npy')
        )
