import argparse
import time

import numpy as np

from nums import numpy as nps
from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.array.base import Block
from nums.core.optimizer.cluster_sim import ClusterState
from nums.core.optimizer.comp_graph import GraphArray
from nums.core.systems.gpu_systems import (
    NumpySerialSystem,
    CupySerialSystem,
    CupyParallelSystem,
)
from nums.core import application_manager as am
from nums.core import settings
from utils import benchmark_func, get_number_of_gpus
import lr_opt as opt
from benchmark_mlp_data import forward, sigmoid_opt, sigmoid_deriv_opt, one_step_fit_np, one_step_fit, np_sample

random_seed = 1337


def distribute_graph_array(app, G, cluster_state):
    for node_id in cluster_state.get_cluster_node_ids():
        for grid_entry in G.grid.get_entry_iterator():
            block: Block = cluster_state.get_block(G.graphs[grid_entry].block_id)
            if node_id not in cluster_state.get_block_node_ids(block.id):
                dst_actor = node_id[0]
                # print(f"dst_actor{dst_actor}")
                app.system.distribute_to(block.oid, dst_actor)  # copy for compute
                cluster_state.commit_copy_block(block.id, node_id)  # copy for optimizer


def one_step_fit_opt_model_parallel(app, X, y, W_in_1, W_1_2, W_2_out, num_gpus, verbose=False):
    # --forward propagation--
    LR = app.one
    cluster_state = ClusterState((num_gpus, 1), app.system)
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)
    X_ga = GraphArray.from_ba(X, cluster_state)
    y_ga = GraphArray.from_ba(y, cluster_state)
    W_in_1_ga = GraphArray.from_ba(W_in_1, cluster_state)
    W_1_2_ga = GraphArray.from_ba(W_1_2, cluster_state)
    W_2_out_ga = GraphArray.from_ba(W_2_out, cluster_state)

    if verbose:
        print("forward Z_1_ga")
    Z_1_ga: GraphArray = forward(app, X_ga, W_in_1_ga)
    if verbose:
        print("forward S_1_ga")
    S_1_ga: GraphArray = sigmoid_opt(app, Z_1_ga, one_ga)
    if verbose:
        print("forward F_1_ga")
    F_1_ga: GraphArray = sigmoid_deriv_opt(app, Z_1_ga, one_ga)
    if verbose:
        print("forward Z_2_ga")
    Z_2_ga: GraphArray = forward(app, S_1_ga, W_1_2_ga)
    S_2_ga: GraphArray = sigmoid_opt(app, Z_2_ga, one_ga)
    F_2_ga: GraphArray = sigmoid_deriv_opt(app, Z_2_ga, one_ga)
    if verbose:
        print("forward Z_out_ga")
    Z_out_ga: GraphArray = forward(app, S_2_ga, W_2_out_ga)
    if verbose:
        print("forward y_predict_ga")
    y_predict_ga: GraphArray = sigmoid_opt(app, Z_out_ga, one_ga)
    if verbose:
        print("forward F_out_ga")
    F_out_ga: GraphArray = sigmoid_deriv_opt(app, Z_out_ga, one_ga)

    # --back propagation--
    if verbose:
        print("collapse D_out_ga")
    D_out_ga = opt.collapse_graph_array(app, F_out_ga.T * (y_predict_ga - y_ga).T)
    if verbose:
        print("collapse D_2_ga")
    D_2_ga = opt.collapse_graph_array(app, F_2_ga.T * (W_2_out_ga @ D_out_ga))
    if verbose:
        print("collapse D_1_ga")
    D_1_ga = opt.collapse_graph_array(app, F_1_ga.T * (W_1_2_ga @ D_2_ga))
    distribute_graph_array(app, D_1_ga, cluster_state)
    if verbose:
        print("collapse_graph_array dW_in_1_ga")
    dW_in_1_ga = opt.collapse_graph_array(app, (D_1_ga @ X_ga).T)
    if verbose:
        print("collapse_graph_array dW_1_2_ga")
    dW_1_2_ga = opt.collapse_graph_array(app, (D_2_ga @ S_1_ga).T)
    if verbose:
        print("collapse_graph_array dW_2_out_ga")
    dW_2_out_ga = opt.collapse_graph_array(app, (D_out_ga @ S_2_ga).T)

    dW_in_1_ga_ba: BlockArray = opt.compute_graph_array(app, dW_in_1_ga)
    dW_1_2_ga_ba: BlockArray = opt.compute_graph_array(app, dW_1_2_ga)
    dW_2_out_ga_ba: BlockArray = opt.compute_graph_array(app, dW_2_out_ga)

    if verbose:
        print("update W_in_1")
    W_in_1 = W_in_1 - LR * dW_in_1_ga_ba
    if verbose:
        print("update W_1_2")
    W_1_2 = W_1_2 - LR * dW_1_2_ga_ba
    if verbose:
        print("update W_2_out")
    W_2_out = W_2_out - LR * dW_2_out_ga_ba

    W_in_1.touch()
    W_1_2.touch()
    W_2_out.touch()


def np_init_weights(app, X, y, d2, dtype):
    dim_1 = 4096  # neurons in the first layer
    dim_2 = d2  # neurons in the second layer

    W_in_1 = app.random.normal(size=(X.shape[1], dim_1)).astype(dtype)
    W_1_2 = app.random.normal(size=(dim_1, dim_2)).astype(dtype)
    W_2_out = app.random.normal(size=(dim_2, y.shape[1])).astype(dtype)
    return W_in_1, W_1_2, W_2_out


def model_parallel_init_weights(app: ArrayApplication, num_gpus, X, y, d2, verbose=False):
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


def benchmark_mlp_model_parallel(num_gpus, N_list, system_class_list, d=140000, optimizer=True, dtype=np.float32):
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
                    W_in_1, W_1_2, W_2_out = model_parallel_init_weights(app, num_gpus, X, y, d2, verbose=False)

                    # Benchmark one step MLP
                    def func():
                        tic = time.time()
                        if optimizer:
                            one_step_fit_opt_model_parallel(app, X, y, W_in_1, W_1_2, W_2_out, num_gpus)
                        else:
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
    optimizer = settings.optimizer
    benchmark_mlp_model_parallel(
        num_gpus,
        N_list=[
            2000,
            # 4096,
            # 8192,
            # 16384,
            # 32768,
            # 70000,
            # 140000,
            # 160000,
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
        optimizer=optimizer,
    )


