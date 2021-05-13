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

from nums.core.array.application import ArrayApplication, BlockArray
from nums.core.array.base import BlockArrayBase

from nums.experimental.optimizer.clusterstate import ClusterState
import nums.experimental.optimizer.grapharray as grapharray
from nums.experimental.optimizer.grapharray import GraphArray, TreeNode, BinaryOp, ReductionOp, Leaf
from nums.experimental.optimizer.tree_search import ExhaustivePlanner, ExhaustiveProcess, Plan

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

def test_serial_no_pruning():
    import conftest 
    grapharray.rop_cls = ReductionOp

    sizes = [2, 3, 4, 5, 6]

    for nblocks in sizes:
        # Square matrix, square cluster shape, block sum problem. 
        # Force blocks to be distributed (cyclically?) over compute nodes:
        # grid shape = k * cluster shape
        # block shape = elementwise matrix shape / grid shape (should work out
        #  to blocksize x blocksize).
        blocksize = 5
        cluster_shape = (blocksize, 1)
        grid_shape = tuple((nblocks, 1))
        matrix_shape = tuple((blocksize*nblocks, blocksize)) 
        block_shape = tuple(np.ceil((np.array(matrix_shape) / np.array(grid_shape))).astype(int).tolist())

        print("cluster shape:", cluster_shape)
        print("matrix shape:", matrix_shape)
        print("grid shape:", grid_shape)
        print("block shape:", block_shape)

        app = conftest.mock_cluster(cluster_shape)
        dtype = np.float64

        print("Cluster up")

        rhs: BlockArray = app.random.random(shape=matrix_shape,
                                      block_shape=block_shape,
                                      dtype=dtype)

        lhs: BlockArray = app.random.random(shape=matrix_shape,
                              block_shape=block_shape,
                              dtype=dtype)
        lhs = lhs.T

        cluster_state: ClusterState = ClusterState(app.cm.devices())
        rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=True)
        lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=True)
        tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=1)
        
        # Check that all resources are equally used to start with
        #mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
        #net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
        #net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
        #assert mem_diff == net_in_diff == net_out_diff == 0

        # Run exhaustive planner, print details of best and worst plans.
        print("Planning")
        serial_planner: ExhaustivePlanner = ExhaustivePlanner(1, unroll=False, prune=False)
        all_plans = serial_planner.solve(tensordot_ga)
        plan: Plan = serial_planner.plan

        print("Number of blocks", nblocks)
        print("Executing plan: ", plan.get_plan_actions())
        print(">>> cost:", plan.get_cost())
        start = time.time()
        result_ga: GraphArray = plan.execute(tensordot_ga)
        end = time.time()
        print("Optimal plan exec time:", end - start)
        # start = time.time()
        # planner.pessimal_plan.execute(block_sum_ga)
        # end = time.time()

        rs = plan.get_cluster_state().resources
        print(">> memory", rs[0])
        print(">> net_in", rs[1])
        print(">> net_out", rs[2])

       # print("Pessimal plan: ", planner.pessimal_plan.get_plan_actions())
       # print(">>> cost:", planner.pessimal_plan.get_cost())
       # print("Pessimal plan exec time:", end - start)
       # pess_rs = planner.pessimal_plan.get_cluster_state().resources
       # print(">> memory", pess_rs[0])
       # print(">> net_in", pess_rs[1])
       # print(">> net_out", pess_rs[2])

        # Print load balancing measures
        mem_diff = max(rs[0]) - min(rs[0])
        net_in_diff = max(rs[1]) - min(rs[1])
        net_out_diff = max(rs[2]) - min(rs[2])

        print("Load imbalance (mem, net in, net out):", mem_diff, net_in_diff, net_out_diff)
        conftest.destroy_mock_cluster(app)


def test_naive_parallel_no_pruning():
    import conftest 
    grapharray.rop_cls = ReductionOp

    runtimes = []
    costs = []
    exec_times = []
    sizes = [2, 3, 4, 5]

    for nblocks in sizes:
        # Square matrix, square cluster shape, block sum problem. 
        # Force blocks to be distributed (cyclically?) over compute nodes:
        # grid shape = k * cluster shape
        # block shape = elementwise matrix shape / grid shape (should work out
        #  to blocksize x blocksize).
        blocksize = 5
        cluster_shape = (1, blocksize)
        grid_shape = tuple((1, nblocks))
        matrix_shape = tuple((blocksize, blocksize*nblocks)) 
        block_shape = tuple(np.ceil((np.array(matrix_shape) / np.array(grid_shape))).astype(int).tolist())

        print("cluster shape:", cluster_shape)
        print("matrix shape:", matrix_shape)
        print("grid shape:", grid_shape)
        print("block shape:", block_shape)

        app = conftest.mock_cluster(cluster_shape)
        dtype = np.float64

        rhs: BlockArray = app.random.random(shape=matrix_shape,
                                      block_shape=block_shape,
                                      dtype=dtype)

        lhs: BlockArray = app.random.random(shape=matrix_shape,
                              block_shape=block_shape,
                              dtype=dtype)
        lhs = lhs.T

        cluster_state: ClusterState = ClusterState(app.cm.devices())
        rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=True)
        lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=True)
        tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=1)

        # Check that all resources are equally used to start with
        #mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
        #net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
        #net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
        #assert mem_diff == net_in_diff == net_out_diff == 0

        # Run exhaustive planner, print details of best and worst plans.
        planner: ExhaustivePlanner = ExhaustivePlanner(16, unroll=False, prune=False)
        all_plans, t_plan = planner.solve(tensordot_ga)
        plan: Plan = planner.plan

        runtimes.append(t_plan)
        costs.append(plan.get_cost())

        print("Number of blocks", nblocks)
        print("Executing plan: ", plan.get_plan_actions())
        print(">>> cost:", plan.get_cost())
        start = time.time()
        result_ga: GraphArray = plan.execute(tensordot_ga)
        end = time.time()
        print("Optimal plan exec time:", end - start)
        exec_times.append(end - start)
        # start = time.time()
        # planner.pessimal_plan.execute(block_sum_ga)
        # end = time.time()

        rs = plan.get_cluster_state().resources
        print(">> memory", rs[0])
        print(">> net_in", rs[1])
        print(">> net_out", rs[2])

       # print("Pessimal plan: ", planner.pessimal_plan.get_plan_actions())
       # print(">>> cost:", planner.pessimal_plan.get_cost())
       # print("Pessimal plan exec time:", end - start)
       # pess_rs = planner.pessimal_plan.get_cluster_state().resources
       # print(">> memory", pess_rs[0])
       # print(">> net_in", pess_rs[1])
       # print(">> net_out", pess_rs[2])

        # Print load balancing measures
        mem_diff = max(rs[0]) - min(rs[0])
        net_in_diff = max(rs[1]) - min(rs[1])
        net_out_diff = max(rs[2]) - min(rs[2])

        print("Load imbalance (mem, net in, net out):", mem_diff, net_in_diff, net_out_diff)
        conftest.destroy_mock_cluster(app)

    for i, s in sizes:
        print(s, "blocks;", runtime[i], "s to plan;", costs[i], "cost;", exec_times[i], "to exec") 


def test_unroll_parallel_no_pruning():
    import conftest 
    grapharray.rop_cls = ReductionOp

    for nblocks in [2, 3, 4, 5, 6]:
        # Square matrix, square cluster shape, block sum problem. 
        # Force blocks to be distributed (cyclically?) over compute nodes:
        # grid shape = k * cluster shape
        # block shape = elementwise matrix shape / grid shape (should work out
        #  to blocksize x blocksize).
        blocksize = 5
        cluster_shape = (1, blocksize)
        grid_shape = tuple((1, nblocks))
        matrix_shape = tuple((blocksize, blocksize*nblocks)) 
        block_shape = tuple(np.ceil((np.array(matrix_shape) / np.array(grid_shape))).astype(int).tolist())

        print("cluster shape:", cluster_shape)
        print("matrix shape:", matrix_shape)
        print("grid shape:", grid_shape)
        print("block shape:", block_shape)

        app = conftest.mock_cluster(cluster_shape)
        dtype = np.float64

        rhs: BlockArray = app.random.random(shape=matrix_shape,
                                      block_shape=block_shape,
                                      dtype=dtype)

        lhs: BlockArray = app.random.random(shape=matrix_shape,
                              block_shape=block_shape,
                              dtype=dtype)
        lhs = lhs.T

        cluster_state: ClusterState = ClusterState(app.cm.devices())
        rhs_ga: GraphArray = GraphArray.from_ba(rhs, cluster_state, copy_on_op=True)
        lhs_ga: GraphArray = GraphArray.from_ba(lhs, cluster_state, copy_on_op=True)
        tensordot_ga = lhs_ga.tensordot(rhs_ga, axes=1)

        # Check that all resources are equally used to start with
        #mem_diff = max(cluster_state.resources[0]) - min(cluster_state.resources[0])
        #net_in_diff = max(cluster_state.resources[1]) - min(cluster_state.resources[1])
        #net_out_diff = max(cluster_state.resources[2]) - min(cluster_state.resources[2])
        #assert mem_diff == net_in_diff == net_out_diff == 0

        # Run exhaustive planner, print details of best and worst plans.
        planner: ExhaustivePlanner = ExhaustivePlanner(16, unroll=True, prune=False)
        all_plans = planner.solve(tensordot_ga)
        plan: Plan = planner.plan

        print("Number of blocks", nblocks)
        print("Executing plan: ", plan.get_plan_actions())
        print(">>> cost:", plan.get_cost())
        start = time.time()
        result_ga: GraphArray = plan.execute(tensordot_ga)
        end = time.time()
        print("Optimal plan exec time:", end - start)
        # start = time.time()
        # planner.pessimal_plan.execute(block_sum_ga)
        # end = time.time()

        rs = plan.get_cluster_state().resources
        print(">> memory", rs[0])
        print(">> net_in", rs[1])
        print(">> net_out", rs[2])

       # print("Pessimal plan: ", planner.pessimal_plan.get_plan_actions())
       # print(">>> cost:", planner.pessimal_plan.get_cost())
       # print("Pessimal plan exec time:", end - start)
       # pess_rs = planner.pessimal_plan.get_cluster_state().resources
       # print(">> memory", pess_rs[0])
       # print(">> net_in", pess_rs[1])
       # print(">> net_out", pess_rs[2])

        # Print load balancing measures
        mem_diff = max(rs[0]) - min(rs[0])
        net_in_diff = max(rs[1]) - min(rs[1])
        net_out_diff = max(rs[2]) - min(rs[2])

        print("Load imbalance (mem, net in, net out):", mem_diff, net_in_diff, net_out_diff)
        conftest.destroy_mock_cluster(app)




if __name__ == "__main__":
    from tests import conftest

    # test_serial_no_pruning()
    test_naive_parallel_no_pruning()
    # test_unroll_parallel_no_pruning()
