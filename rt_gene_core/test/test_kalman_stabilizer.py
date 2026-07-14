import numpy as np

from rt_gene.kalman_stabilizer import Stabilizer


def test_scalar_stabilizer_accepts_numpy_float_measurements():
    stabilizer = Stabilizer(state_num=2, measure_num=1)

    stabilizer.update([np.float64(0.1)])
    stabilizer.update([np.float32(0.2)])

    assert stabilizer.state.dtype == np.float32


def test_point_stabilizer_accepts_numpy_float_measurements():
    stabilizer = Stabilizer(state_num=4, measure_num=2)

    stabilizer.update([np.float64(0.1), np.float32(0.2)])

    assert stabilizer.state.dtype == np.float32
