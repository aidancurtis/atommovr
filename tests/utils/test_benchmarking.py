import numpy as np

import atommover.algorithms as algos
from atommover.utils.benchmarking import Benchmarking
from atommover.utils.errormodels import UniformVacuumTweezerError


def test_run():
    algo = algos.Hungarian()
    error_models = [
        UniformVacuumTweezerError(pickup_fail_rate=0.001, putdown_fail_rate=0.001)
    ]
    sys_sizes = list(range(10, 14))
    bench = Benchmarking(
        algos=[algo],
        sys_sizes=sys_sizes,
        error_models_list=error_models,
        rounds_list=[1],
        n_shots=100,
    )
    bench.run()
