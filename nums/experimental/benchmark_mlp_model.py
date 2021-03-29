import argparse
import time

import numpy as np
import ray

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.base import Block
from nums.core.systems.gpu_systems import (
    NumpySerialSystem,
    CupySerialSystem,
    CupyParallelSystem,
)
from nums.core import application_manager as am
from nums.core import settings
from utils import benchmark_func, get_number_of_gpus
from benchmark_mlp_data import one_step_fit_np, one_step_fit, np_sample


random_seed = 1337


def np_init_weights(app, X, y, d2, dtype):
    dim_1 = 4096  # neurons in the first layer
    dim_2 = d2  # neurons in the second layer

    W_in_1 = app.random.normal(size=(X.shape[1], dim_1)).astype(dtype)
    W_1_2 = app.random.normal(size=(dim_1, dim_2)).astype(dtype)
    W_2_out = app.random.normal(size=(dim_2, y.shape[1])).astype(dtype)
    return W_in_1, W_1_2, W_2_out


def model_init_weights(app: ArrayApplication, num_gpus, X, y, d2, verbose=False):
    dim_1 = 4096  # neurons in the first layer
    dim_2 = d2  # neurons in the second layer
    W_in_1 = app.random.normal(shape=(X.shape[1], dim_1), block_shape=(X.shape[1] // num_gpus, dim_1), dtype=X.dtype)
    W_1_2 = app.random.normal(shape=(dim_1, dim_2), block_shape=(dim_1, dim_2 // num_gpus), dtype=X.dtype)
    W_2_out = app.random.normal(shape=(dim_2, y.shape[1]), block_shape=(dim_2 // num_gpus, y.block_shape[1]),
                                    dtype=X.dtype)
    if verbose:
        print(f"W_in_1.shape {W_in_1.shape} W_in_1.block_shape {W_in_1.block_shape}")
        print(f"W_1_2.shape {W_1_2.shape} W_1_2.block_shape {W_1_2.block_shape}")
        print(f"W_2_out.shape {W_2_out.shape} W_2_out.block_shape {W_2_out.block_shape}")
    return W_in_1, W_1_2, W_2_out


def sample(app: ArrayApplication, sample_size, feature, num_gpus, dtype):  
    X_train = app.random.normal(shape=(sample_size, feature), block_shape=(sample_size, feature // num_gpus),
                                    dtype=dtype)
    y_train = app.ones(shape=(sample_size, 1), block_shape=(sample_size, 1), dtype=dtype)
    return X_train, y_train


def benchmark_mlp_model(num_gpus, N_list, system_class_list, d=140000, dtype=np.float32):
    format_string = "%20s,%10s,%10s,%10s,%10s,%10s"
    print(format_string % ("Library", "N", "d_in", "d_2", "Cost", "CV"))

    for N in N_list:
        N = int(N)
        d2 = 20000
        for system_class in system_class_list:
            # try:
            if True:
                if system_class in ["Cupy", "Numpy"]:
                    name = system_class
                    import cupy as cp

                    arr_lib = cp if system_class == "Cupy" else np
                    arr_lib.inv = arr_lib.linalg.inv
                    app = arr_lib

                    X, y = np_sample(np, sample_size=N, feature=d, dtype=dtype)
                    W_in_1, W_1_2, W_2_out = np_init_weights(np, X, y, d2, dtype=dtype)

                    X = cp.asarray(X)
                    y = cp.asarray(y)
                    W_in_1 = cp.asarray(W_in_1)
                    W_1_2 = cp.asarray(W_1_2)
                    W_2_out = cp.asarray(W_2_out)

                    cp.cuda.Device(0).synchronize()

                    # Benchmark one step mlp
                    def func():
                        tic = time.time()
                        one_step_fit_np(arr_lib, X, y, W_in_1, W_1_2, W_2_out)
                        cp.cuda.Device(0).synchronize()
                        toc = time.time()
                        return toc - tic, None

                    costs = benchmark_func(func)
                    del (X, y, W_in_1, W_1_2, W_2_out)
                else:
                    # Init system
                    name = system_class.__name__
                    app = am.instance()

                    # Make dataset
                    nps.random.seed(0)
                    X, y = sample(app, sample_size=N, feature=d, num_gpus=num_gpus, dtype=dtype)
                    W_in_1, W_1_2, W_2_out = model_init_weights(app, num_gpus, X, y, d2, verbose=False)

                    # Benchmark one step MLP
                    def func():
                        tic = time.time()
                        one_step_fit(app, X, y, W_in_1, W_1_2, W_2_out)
                        toc = time.time()
                        return toc - tic, None

                    costs = benchmark_func(func)

                    del (X, y, app, W_in_1, W_1_2, W_2_out)
            # except Exception:
            else:
                costs = [-1]

            log_str = format_string % (
                name,
                "%d" % N,
                "%d" % d,
                "%d" % d2,
                "%.4f" % np.mean(costs),
                "%.2f" % (np.std(costs) / np.mean(costs)),
            )
            print(log_str)
            with open("result_mlp_model.csv", "a") as f:
                f.write(log_str + "\n")


if __name__ == "__main__":
    num_gpus = settings.num_gpus

    benchmark_mlp_model(
        num_gpus,
        N_list=[
            # 2000,
            4096,
            8192,
            16384,
            32768,
            70000,
            140000,
            160000,
            # 3000,
            # 0.5e6 / 4,
            # 1e6 / 4,
            # 2e6 / 4,
            # 3e6 / 4,
            # 5e6 / 4,
            # 10e6 / 4,
            # 20e6 / 4,
            # 40e6 / 4,
            # 80e6 / 4,
            # 160e6 / 4,
            # 200e6 / 4,
        ],
        system_class_list=[
            "Cupy",
            CupyParallelSystem,
            # "Numpy",
        ],
    )


