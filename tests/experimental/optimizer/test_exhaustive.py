# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import itertools
import time

import numpy as np
import dill

from nums.core.systems.systems import System, SerialSystem, RaySystem
from nums.core.systems.schedulers import RayScheduler, TaskScheduler, BlockCyclicScheduler
from nums.core.array.application import ArrayApplication, BlockArray
from nums.core.array.base import BlockArrayBase

from nums.experimental.optimizer.cluster_sim import ClusterState
from nums.experimental.optimizer.comp_graph import GraphArray, TreeNode, BinaryOp, ReductionOp, Leaf
from nums.experimental.optimizer.tree_search import RandomTS, BlockCyclicTS, ExhaustivePlanner, ExhaustiveProcess, Plan
import common


def optimized_tensordot(lhs: BlockArrayBase, rhs: BlockArrayBase, axes,
                        copy_on_op=True) -> BlockArray:
    system: System = lhs.system
    if isinstance(system, RaySystem) and isinstance(system.scheduler, BlockCyclicScheduler):
        cluster_state = ClusterState(system.scheduler.cluster_shape, system)
    else:
        cluster_state = ClusterState((1,), system)
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=copy_on_op)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=copy_on_op)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)
    global random_state
    print("op grid shape", tensordot_ga.grid.grid_shape)

    planner: ExhaustivePlanner = ExhaustivePlanner(1)
    planner.solve(tensordot_ga)
    plan: Plan = planner.plan
    result_ga: GraphArray = plan.execute(tensordot_ga)

    print("mem", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])
    return BlockArray(result_ga.grid, system, result_ga.to_blocks())


def traverse(ga: GraphArray):
    visited = set()
    root = ga.graphs[0][0]
    print("graphs type", type(ga.graphs), type(ga.graphs[0]), type(ga.graphs[0][0]))
    remaining = [root]
    while len(remaining) > 0:
        current = remaining.pop(0)
        print(current)
        print(type(current))
        visited.add(str(current.tree_node_id))
        for c in current.get_children():
            if c in visited:
                continue
            remaining.append(c)


def test_matvec(app_inst: ArrayApplication):
    A_shape, A_block_shape = (5, 10), (5, 5)
    x_shape, x_block_shape = (10, 1), (5, 1)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    real_x = np.random.random(np.product(x_shape)).reshape(x_shape)
    A: BlockArray = app_inst.array(real_A, A_block_shape)
    x: BlockArray = app_inst.array(real_x, x_block_shape)
    result: BlockArray = A @ x
    opt_result: BlockArray = optimized_tensordot(A, x, axes=1)
    assert np.allclose(result.get(), real_A @ real_x)
    assert app_inst.allclose(result, opt_result).get()


def test_matmat(app_inst: ArrayApplication):
    X_shape, X_block_shape = (5, 10), (5, 5)
    Y_shape, Y_block_shape = (10, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)
    Z: BlockArray = X @ Y
    opt_Z: BlockArray = optimized_tensordot(X, Y, axes=1)
    assert np.allclose(Z.get(), real_X @ real_Y)
    assert app_inst.allclose(Z, opt_Z).get()


def test_big_matmat(app_inst: ArrayApplication):
    num_blocks = 10**3
    X_shape, X_block_shape = (5, 5*num_blocks), (5, 5)
    Y_shape, Y_block_shape = (5*num_blocks, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)
    Z: BlockArray = X @ Y
    t = time.time()
    opt_Z: BlockArray = optimized_tensordot(X, Y, axes=1)
    print(time.time()-t)
    assert np.allclose(Z.get(), real_X @ real_Y)
    assert app_inst.allclose(Z, opt_Z).get()


def test_load_sqr():
    # Set up test: 5 compute nodes, 5 5x5 blocks in X and in Y
    # Check that regular-path NumS gets the same result as NumPy
    #   for X @ Y.
    num_nodes = 5
    app_inst = common.mock_cluster((num_nodes, 1))
    num_blocks = 5
    X_shape, X_block_shape = (5*num_blocks, 5), (5, 5)
    Y_shape, Y_block_shape = (5*num_blocks, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)
    result: BlockArray = X.T @ Y
    assert np.allclose(result.get(), real_X.T @ real_Y)

    # Set up system and initialize GraphArray versions of X, Y, and X@Y.
    lhs, rhs, axes = X.T, Y, 1
    system: System = lhs.system
    if isinstance(system, RaySystem) and isinstance(system.scheduler, BlockCyclicScheduler):
        print("ray system, block cyclic")
        cluster_state: ClusterState = ClusterState(system.scheduler.cluster_shape, system)
    else:
        cluster_state: ClusterState = ClusterState((1,), system)
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=True)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=True)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)

    # Check that all resources are equally used to start with
    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    assert mem_diff == net_in_diff == net_out_diff == 0
    # Block-cyclic distribution of 100 blocks of size 25 over 10 nodes == 10*25 == 250
    # We have 2 such matrices, so expect initial memory to be 500.
    print(cluster_state.resources)
    # assert max(cluster_state.resources[0]) == (num_blocks/num_nodes)*25*2
    assert max(cluster_state.resources[1]) == max(cluster_state.resources[2]) == 0

    # Run exhaustive planner, print details of best and worst plans.
    planner: ExhaustivePlanner = ExhaustivePlanner(4)
    all_plans = planner.solve(tensordot_ga)
    plan: Plan = planner.plan

    print("Executing plan: ", plan.get_plan_actions())
    print(">>> cost:", plan.get_cost())
    start = time.time()
    result_ga: GraphArray = plan.execute(tensordot_ga)
    end = time.time()
    print("Optimal plan exec time:", end - start)
    start = time.time()
    planner.pessimal_plan.execute(tensordot_ga)
    end = time.time()
    opt_result = BlockArray(result_ga.grid, system, result_ga.to_blocks())
    assert app_inst.allclose(result, opt_result).get()

    rs = plan.get_cluster_state().resources
    print(">> memory", rs[0])
    print(">> net_in", rs[1])
    print(">> net_out", rs[2])

    print("Pessimal plan: ", planner.pessimal_plan.get_plan_actions())
    print(">>> cost:", planner.pessimal_plan.get_cost())
    print("Pessimal plan exec time:", end - start)
    pess_rs = planner.pessimal_plan.get_cluster_state().resources
    print(">> memory", pess_rs[0])
    print(">> net_in", pess_rs[1])
    print(">> net_out", pess_rs[2])

    # Check that each plan has the right operators and that a min-cost
    # plan was chosen.
    print("Checking that min cost plan chosen, all plans have right ops")
    min_cost_plans = 0
    for p, c in all_plans:
        assert plan.get_cost() <= c
        if plan.get_cost() == c:
            min_cost_plans += 1
        actions = p.get_plan_actions()
        bop_mult = 0
        bop_add = 0
        reduc_add = 0
        for a in actions:
            if isinstance(a, BinaryOp) and a.op_name == "tensordot":
                bop_mult += 1
            elif isinstance(a, BinaryOp) and a.op_name == "add":
                bop_add += 1
            elif isinstance(a, ReductionOp) and a.op_name == "add":
                reduc_add += 1
        assert bop_mult == 5
        assert bop_add == 1
        assert reduc_add == 3
    print("Total plans:", len(all_plans))
    print("Total min-cost plans:", min_cost_plans)

    # Print load balancing measures
    mem_diff = max(rs[0]) - min(rs[0])
    net_in_diff = max(rs[1]) - min(rs[1])
    net_out_diff = max(rs[2]) - min(rs[2])

    print("Load imbalance (mem, net in, net out):", mem_diff, net_in_diff, net_out_diff)


def test_load_single_block_rhs():
    app_inst = common.mock_cluster((10, 1))
    num_blocks = 100
    X_shape, X_block_shape = (5*num_blocks, 5), (5, 5)
    Y_shape, Y_block_shape = (5, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)

    lhs, rhs, axes = X, Y, 1
    system: System = lhs.system
    if isinstance(system, RaySystem) and isinstance(system.scheduler, BlockCyclicScheduler):
        cluster_state: ClusterState = ClusterState(system.scheduler.cluster_shape, system)
    else:
        cluster_state: ClusterState = ClusterState((1,), system)
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)

    print("memory", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])
    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    print(mem_diff, net_in_diff, net_out_diff)
    assert mem_diff == 25  # b/c single block array is placed in node 1.
    assert net_in_diff == net_out_diff == 0
    assert max(cluster_state.resources[1]) == max(cluster_state.resources[2]) == 0
    result_ga: GraphArray = RandomTS(
        seed=np.random.RandomState(1337),
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(tensordot_ga)
    print("memory", cluster_state.resources[0])
    print("net_in", cluster_state.resources[1])
    print("net_out", cluster_state.resources[2])

    mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
    net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
    net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
    print(mem_diff, net_in_diff, net_out_diff)
    # assert mem_diff <= 25 and net_in_diff <= 50 and net_out_diff <= 250


def test_save_to_file():
    # Set up test: 5 compute nodes, 5 5x5 blocks in X and in Y
    # Check that regular-path NumS gets the same result as NumPy
    #   for X @ Y.
    num_nodes = 1
    app_inst = common.mock_cluster((num_nodes, 1))
    num_blocks = 1
    X_shape, X_block_shape = (5*num_blocks, 5), (5, 5)
    Y_shape, Y_block_shape = (5*num_blocks, 5), (5, 5)
    real_X = np.random.random(np.product(X_shape)).reshape(X_shape)
    real_Y = np.random.random(np.product(Y_shape)).reshape(Y_shape)
    X: BlockArray = app_inst.array(real_X, X_block_shape)
    Y: BlockArray = app_inst.array(real_Y, Y_block_shape)
    result: BlockArray = X.T @ Y
    assert np.allclose(result.get(), real_X.T @ real_Y)

    # Set up system and initialize GraphArray versions of X, Y, and X@Y.
    lhs, rhs, axes = X.T, Y, 1
    system: System = lhs.system
    if isinstance(system, RaySystem) and isinstance(system.scheduler, BlockCyclicScheduler):
        cluster_state: ClusterState = ClusterState(system.scheduler.cluster_shape, system)
    else:
        cluster_state: ClusterState = ClusterState((1,), system)
    lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=True)
    rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=True)
    tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=axes)
    assert max(cluster_state.resources[1]) == max(cluster_state.resources[2]) == 0

    # Run exhaustive planner.
    planner: ExhaustivePlanner = ExhaustivePlanner(1)
    all_plans = planner.solve(tensordot_ga)
    plan: Plan = planner.plan

    filename = "/tmp/plan.pkl"
    planner.serialize(pessimal=False, filename=filename)
    revived_plan: Plan = None
    with open(filename, "rb") as f:
        revived_plan = dill.load(f)

    assert revived_plan.cost == plan.cost
    assert len(plan.plan) == len(revived_plan.plan)
    for i, step in enumerate(plan.plan):
        assert step == revived_plan.plan[i]


if __name__ == "__main__":
    from tests import conftest

    app_inst = conftest.get_app("ray-cyclic")
#    test_matvec(app_inst)
#    test_matmat(app_inst)
#    test_big_matmat(app_inst)
    test_load_sqr()
#    test_save_to_file()
#    test_load_single_block_rhs()
