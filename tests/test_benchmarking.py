import numpy as np

import atommover.algorithms as algos
from atommover.benchmarking import Benchmarking


def test_run():
    algo = algos.Hungarian()
    sys_sizes = list(range(10, 14))
    bench = Benchmarking(
        algos=[algo],
        sys_sizes=sys_sizes,
        rounds_list=[1],
        n_shots=100,
    )
    bench.run()
