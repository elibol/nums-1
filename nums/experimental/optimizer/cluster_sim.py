# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools

import numpy as np
from nums.core.array.base import Block
from nums.core.systems.systems import System


class ClusterState(object):

    def __init__(self, cluster_shape, system: System):
        self.cluster_shape = cluster_shape
        self.system = system
        # 3 matrices: mem, net_in, net_out.
        self.mem_idx, self.net_in_idx, self.net_out_idx = 0, 1, 2
        self.resources: np.ndarray = np.zeros(shape=tuple([3]+list(self.cluster_shape)),
                                              dtype=np.float)
        # Dict from block id to Block.
        self.blocks = {}
        # Dict from block id to list of node id.
        self.block_nodes = {}

    def copy(self):
        new_cluster = ClusterState(self.cluster_shape, self.system)
        # Copy nodes.
        new_cluster.resources = self.resources.copy()
        # Copy blocks.
        for block_id in self.blocks:
            # Don't copy blocks themselves. Updating references is enough.
            new_cluster.blocks[block_id] = self.blocks[block_id]
            new_cluster.block_nodes[block_id] = list(self.block_nodes[block_id])
        return new_cluster

    def get_cluster_node_ids(self):
        return list(self.get_cluster_entry_iterator())

    def get_cluster_entry_iterator(self):
        return itertools.product(*map(range, self.cluster_shape))

    def get_cluster_entry(self, grid_entry):
        cluster_entry = []
        num_grid_entry_axes = len(grid_entry)
        num_cluster_axes = len(self.cluster_shape)
        if num_grid_entry_axes <= num_cluster_axes:
            # When array has fewer or equal # of axes than cluster.
            for cluster_axis in range(num_cluster_axes):
                if cluster_axis < num_grid_entry_axes:
                    cluster_dim = self.cluster_shape[cluster_axis]
                    grid_entry_dim = grid_entry[cluster_axis]
                    cluster_entry.append(grid_entry_dim % cluster_dim)
                else:
                    cluster_entry.append(0)
        elif num_grid_entry_axes > num_cluster_axes:
            # When array has more axes then cluster.
            for cluster_axis in range(num_cluster_axes):
                cluster_dim = self.cluster_shape[cluster_axis]
                grid_entry_dim = grid_entry[cluster_axis]
                cluster_entry.append(grid_entry_dim % cluster_dim)
            # Ignore trailing axes, as these are "cycled" to 0 by assuming
            # the dimension of those cluster axes is 1.
        return tuple(cluster_entry)

    # Block Ops.

    def add_block(self, block: Block, node_ids):
        assert block.id not in self.blocks and block.id not in self.block_nodes
        self.blocks[block.id] = block
        self.block_nodes[block.id] = node_ids

    def get_block(self, block_id) -> Block:
        return self.blocks[block_id]

    def get_block_node_ids(self, block_id):
        return self.block_nodes[block_id]

    def get_block_node_id(self, block_id):
        # The most recent node to which this object was transferred.
        return self.get_block_node_ids(block_id)[-1]

    def union_nodes(self, block_id_a, block_id_b):
        block_a_node_ids = self.get_block_node_ids(block_id_a)
        block_b_node_ids = self.get_block_node_ids(block_id_b)
        return list(set(block_a_node_ids).union(set(block_b_node_ids)))

    def mutual_nodes(self, block_id_a, block_id_b):
        block_a_node_ids = self.get_block_node_ids(block_id_a)
        block_b_node_ids = self.get_block_node_ids(block_id_b)
        return list(set(block_a_node_ids).intersection(set(block_b_node_ids)))

    def blocks_local(self, block_id_a, block_id_b):
        return len(self.mutual_nodes(block_id_a, block_id_b)) > 0

    def init_mem_load(self, node_id, block_id):
        block: Block = self.get_block(block_id)
        block_node_ids: list = self.get_block_node_ids(block_id)
        assert node_id in block_node_ids
        size = block.size()
        self.resources[self.mem_idx][node_id] += size

    def simulate_copy_block(self, block_id, to_node_id, resources):
        block: Block = self.get_block(block_id)
        block_node_ids: list = self.get_block_node_ids(block_id)
        if to_node_id in block_node_ids:
            return resources
        # Pick the first node. This is the worst-case assumption,
        # since it imposes the greatest load (w.r.t. cost function) on the network,
        # though we really don't have control over this.
        from_node_id = block_node_ids[0]
        # Update load.
        size = block.size()
        resources[self.net_out_idx][from_node_id] += size
        resources[self.net_in_idx][to_node_id] += size
        resources[self.mem_idx][to_node_id] += size
        return resources

    def simulate_op(self, op_mem: int, block_id_a, block_id_b, node_id, resources):
        if node_id not in self.get_block_node_ids(block_id_a):
            resources = self.simulate_copy_block(block_id_a, node_id, resources)
        if node_id not in self.get_block_node_ids(block_id_b):
            resources = self.simulate_copy_block(block_id_b, node_id, resources)
        resources[self.mem_idx][node_id] += op_mem
        return resources

    def commit_copy_block(self, block_id, to_node_id):
        self.resources = self.simulate_copy_block(block_id, to_node_id, self.resources)
        # Update node location.
        block_node_ids: list = self.get_block_node_ids(block_id)
        block_node_ids.append(to_node_id)

    def commit_op(self, op_mem: int, block_id_a, block_id_b, node_id):
        if node_id not in self.get_block_node_ids(block_id_a):
            self.commit_copy_block(block_id_a, node_id)
        if node_id not in self.get_block_node_ids(block_id_b):
            self.commit_copy_block(block_id_b, node_id)
        self.resources[self.mem_idx][node_id] += op_mem

    def simulate_uop(self, op_mem: int, block_id, node_id, resources):
        if node_id not in self.get_block_node_ids(block_id):
            resources = self.simulate_copy_block(block_id, node_id, resources)
        resources[self.mem_idx][node_id] += op_mem
        return resources

    def commit_uop(self, op_mem: int, block_id, node_id):
        if node_id not in self.get_block_node_ids(block_id):
            self.commit_copy_block(block_id, node_id)
        self.resources[self.mem_idx][node_id] += op_mem
