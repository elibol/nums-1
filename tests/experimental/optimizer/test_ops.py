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

from nums.core.array.application import ArrayApplication, BlockArray
from nums.experimental.optimizer.cluster_sim import ClusterState
from nums.experimental.optimizer.comp_graph import GraphArray, TreeNode, BinaryOp, ReductionOp, Leaf
from nums.experimental.optimizer.tree_search import RandomTS, BlockCyclicTS
import common


def test_neg():
    cluster_shape = (4, 1)
    app: ArrayApplication = common.mock_cluster(cluster_shape)
    cluster_state = ClusterState(cluster_shape, app.system)

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = -A_ga
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.system, result_ga.to_blocks())
    assert app.allclose(-A, result_ba)


def test_root_uop():
    cluster_shape = (4, 1)
    app: ArrayApplication = common.mock_cluster(cluster_shape)
    cluster_state = ClusterState(cluster_shape, app.system)

    one_ba: BlockArray = app.one
    one_ga: GraphArray = GraphArray.from_ba(app.one, cluster_state)

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = one_ga / (one_ga + app.exp(-(A_ga + A_ga)))
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.system, result_ga.to_blocks())
    print(app.allclose(result_ba, one_ba / (one_ba + app.exp(-(A + A)))).get())
    assert app.allclose(result_ba, one_ba / (one_ba + app.exp(-(A + A))))


def test_transpose():
    cluster_shape = (4, 1)
    app: ArrayApplication = common.mock_cluster(cluster_shape)
    cluster_state = ClusterState(cluster_shape, app.system)

    A_shape, A_block_shape = (5, 10), (5, 5)
    real_A = np.random.random(np.product(A_shape)).reshape(A_shape)
    A: BlockArray = app.array(real_A, A_block_shape)
    A_ga: GraphArray = GraphArray.from_ba(A, cluster_state)
    prob_ga: GraphArray = A_ga.T @ A_ga
    result_ga: GraphArray = RandomTS(
        seed=1337,
        max_samples_per_step=1,
        max_reduction_pairs=1,
        force_final_action=True).solve(prob_ga)
    result_ba = BlockArray(result_ga.grid, app.system, result_ga.to_blocks())
    assert app.allclose(A.T @ A, result_ba)


if __name__ == "__main__":
    from tests import conftest

    test_neg()
    test_root_uop()
    test_transpose()
