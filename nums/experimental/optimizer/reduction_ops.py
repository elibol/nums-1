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
import copy

import numpy as np
import scipy.special

from nums.core.storage.storage import ArrayGrid
from nums.core.array.base import BlockArrayBase, Block
from nums.core.array import utils as array_utils
from nums.experimental.optimizer.clusterstate import ClusterState
from nums.experimental.optimizer.graph import TreeNode, Leaf


def subsample(total_items, max_items, rs: np.random.RandomState):
    perms = rs.permutation(total_items)
    if total_items < max_items:
        return perms
    return perms[:max_items]


class ReductionOp(TreeNode):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None, seed=1337):
        super().__init__(cluster_state, tree_node_id)
        self.op_name = None
        # For sampling pairs of leafs in get_actions.
        self.rs = np.random.RandomState(seed)
        self.children_dict: dict = {}
        self.leafs_dict: dict = {}

    def __repr__(self):
        return "Reduc(id=%s, op=%s, in=%d)" % (str(self.tree_node_id),
                                               self.op_name,
                                               len(self.children_dict))

    def get_children(self):
        return [self.children_dict[key] for key in sorted(self.children_dict.keys())]

    def num_nodes(self):
        r = 1
        for _, child in self.children_dict.items():
            r += child.num_nodes()
        return r

    def copy(self, cluster_state, parent=None, new_ids=False):
        rop: ReductionOp = ReductionOp(cluster_state, None if new_ids else self.tree_node_id)
        assert (rop.tree_node_id is not None
                and (new_ids or rop.tree_node_id == self.tree_node_id))
        rop.parent = parent
        rop.op_name = self.op_name
        rop.copy_on_op = self.copy_on_op
        for child_id, child in self.children_dict.items():
            child_copy: TreeNode = child.copy(cluster_state=cluster_state,
                                              parent=rop,
                                              new_ids=new_ids)
            assert (child_copy.tree_node_id is not None
                    and (new_ids or child_copy.tree_node_id == child_id))
            rop.children_dict[child_copy.tree_node_id] = child_copy
            if child.tree_node_id in self.leafs_dict:
                rop.leafs_dict[child_copy.tree_node_id] = child_copy
        # TODO (hme): How do we properly copy random state?
        return rop

    def add_child(self, child: TreeNode):
        assert child not in self.children_dict
        self.children_dict[child.tree_node_id] = child
        if isinstance(child, Leaf):
            self.leafs_dict[child.tree_node_id] = child

    def test_integrity(self):
        # This is expensive and only used for testing.
        for leaf_id, leaf in self.leafs_dict.items():
            assert leaf_id == leaf.tree_node_id
        for child_id, child in self.children_dict.items():
            assert child_id == child.tree_node_id
            if isinstance(child, Leaf):
                assert child.tree_node_id in self.leafs_dict

    def update_child(self, old_children, new_children):
        # TODO: Remove integrity checks.
        # self.test_integrity()
        for old_child in old_children:
            assert old_child.tree_node_id in self.children_dict, "Failed to update child: Old " \
                                                                 "child isn't a child of this node."
            del self.children_dict[old_child.tree_node_id]
            if old_child.tree_node_id in self.leafs_dict:
                del self.leafs_dict[old_child.tree_node_id]
        for new_child in new_children:
            self.children_dict[new_child.tree_node_id] = new_child
            if isinstance(new_child, Leaf):
                self.leafs_dict[new_child.tree_node_id] = new_child
        # self.test_integrity()

    def get_leafs(self):
        leafs = []
        for child_id, child in self.children_dict.items():
            leafs += child.get_leafs()
        return leafs

    def is_frontier(self):
        # This is a frontier if all children are computed.
        # This is a stronger constraint than just 2 leafs, but allows
        # for better pairing of operations during action selction.
        return len(self.leafs_dict) == len(self.children_dict)

    def get_frontier(self):
        # This poses an interesting generalization to our prior assumptions about frontiers.
        # We can now have this node be a frontier, as there are actions we can perform on it.
        # It may also contain children that are also frontiers, so collect those.
        # We generate the set of actions from these frontier nodes using their
        # respective actions methods.
        frontier_nodes = []
        if self.is_frontier():
            frontier_nodes.append(self)
        for child_id, child in self.children_dict.items():
            frontier_nodes += child.get_frontier()
        return frontier_nodes

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a function. Second entry is kwargs.
        Invoked actions return a new node without mutating the tree,
        which is always a leaf for BinaryOp.
        """
        actions = []
        if self.is_frontier():
            unique_reduction_pairs = kwargs.get("unique_reduction_pairs", None)
            max_pairs = kwargs.get("max_reduction_pairs", False)
            num_leafs = len(self.leafs_dict)
            if num_leafs == 2:
                leaf_id_pairs = [tuple(self.leafs_dict.keys())]
            elif unique_reduction_pairs:
                # Do a random pairing of all leafs.
                immediate_leaf_ids = list(self.leafs_dict.keys())
                idx_pool = self.rs.permutation(len(immediate_leaf_ids))
                if len(idx_pool) % 2 == 1:
                    idx_pool = idx_pool[:-1]
                leaf_id_pairs = []
                for i in range(0, len(idx_pool), 2):
                    leaf_id_pairs.append(idx_pool[i:i + 2])
            elif max_pairs is not None:
                # This can be optimized further.
                num_pairs = scipy.special.binom(len(self.leafs_dict), 2)
                immediate_leaf_ids = list(self.leafs_dict.keys())
                if num_pairs <= max_pairs:
                    leaf_id_pairs = list(itertools.combinations(immediate_leaf_ids, r=2))
                elif max_pairs <= num_pairs // 2:
                    # This will sample faster for small max_pairs.
                    leaf_pair_set = set()
                    leaf_id_pairs = []
                    for _ in range(max_pairs):
                        idx_pair = tuple(self.rs.randint(0, num_leafs, 2))
                        while idx_pair[0] == idx_pair[1] or idx_pair in leaf_pair_set:
                            idx_pair = tuple(self.rs.randint(0, num_leafs, 2))
                        leaf_pair_set.add(idx_pair)
                        leaf_id_pairs.append((immediate_leaf_ids[idx_pair[0]],
                                              immediate_leaf_ids[idx_pair[1]]))
                else:
                    a_idxs = self.rs.permutation(len(immediate_leaf_ids))
                    b_idxs = self.rs.permutation(len(immediate_leaf_ids))
                    leaf_id_pairs = set()
                    while len(leaf_id_pairs) < max_pairs:
                        for a_idx in a_idxs:
                            for b_idx in b_idxs:
                                if a_idx == b_idx:
                                    continue
                                pair = immediate_leaf_ids[a_idx], immediate_leaf_ids[b_idx]
                                if pair not in leaf_id_pairs:
                                    leaf_id_pairs.add(pair)
                                    break
                            if len(leaf_id_pairs) >= max_pairs:
                                break
                    leaf_id_pairs = list(leaf_id_pairs)
            else:
                # This grows exponentially w/ number of leafs.
                leaf_id_pairs = list(itertools.combinations(list(self.leafs_dict.keys()), r=2))

            use_all_nodes = kwargs.get("use_all_nodes", False)
            for leaf_id_pair in leaf_id_pairs:
                assert leaf_id_pair[0] != leaf_id_pair[1]
                if use_all_nodes:
                    node_ids = self.cluster_state.get_cluster_node_ids()
                else:
                    # Restrict node ids to the nodes on which the leafs already reside.
                    node_ids = self.cluster_state.union_nodes(
                        self.leafs_dict[leaf_id_pair[0]].block_id,
                        self.leafs_dict[leaf_id_pair[1]].block_id)
                for node_id in node_ids:
                    actions.append((self.tree_node_id, {"node_id": node_id,
                                                        "leaf_ids": leaf_id_pair}))
        return actions

    def simulate_on(self, node_id, leaf_ids=None) -> np.ndarray:
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_op(self._mem_cost(leafs),
                                                   left.block_id,
                                                   right.block_id,
                                                   node_id,
                                                   resources)
        return resources

    def execute_on(self, node_id, leaf_ids=None, plan_only=False) -> TreeNode:
        """
        This can return:
        - Another ReductionOp.
        - A BinaryOp.
        """
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        result = self._collapse(node_id, left, right, plan_only=plan_only)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_op(self._mem_cost(leafs),
                                     left.block_id,
                                     right.block_id,
                                     node_id)
        # Update cluster state with new block.
        self.cluster_state.add_block(new_block, [node_id])
        assert self.cluster_state.blocks_local(left.block_id, right.block_id)
        assert self.cluster_state.blocks_local(left.block_id, new_leaf.block_id)
        # The following are mutating operations.
        # Set the new leaf's parent to this node.
        new_leaf.parent = self
        # Update this node's children: We've collapsed two child leafs by performing
        # the reduction operation, so remove those leafs and replace them with the new leaf.
        self.update_child(leafs, [new_leaf])
        if len(self.children_dict) == 1:
            assert tuple(self.children_dict.values())[0] is new_leaf
            # This was constructed as a reduction with two children,
            # otherwise the reduction would have been transformed into a binary op.
            # We can return the leaf,
            # but we need to perform some mutations to remove this node from the graph.
            # Remove the node from parent reference.
            if self.parent is not None:
                self.parent.update_child([self], [new_leaf])
            # Remove the node as new_leaf's parent.
            new_leaf.parent = self.parent
            return new_leaf
        else:
            return self

    def _collapse(self, node_id, left: Leaf, right: Leaf, plan_only):
        lblock: Block = self.cluster_state.get_block(left.block_id)
        rblock: Block = self.cluster_state.get_block(right.block_id)
        if self.op_name == "matmul":
            op_name, args = "tensordot", {"axes": 1}
            assert lblock.shape[1] == rblock.shape[0]
        else:
            op_name, args = self.op_name, {}
            assert lblock.shape == rblock.shape
        options: dict = self.cluster_state.system.get_options(node_id,
                                                              self.cluster_state.cluster_shape)
        block: Block = lblock.bop(op_name, rblock, args=args, options=options, plan_only=plan_only)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block_id = block.id
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self, leafs):
        # Computes the memory required to perform this operation.
        # We approximate by just computing the memory required to store the result.
        assert leafs is not None and len(leafs) > 0
        shape = None
        for leaf in leafs:
            assert leaf.tree_node_id in self.leafs_dict
            leaf_block: Block = self.cluster_state.get_block(leaf.block_id)
            if shape is None:
                shape = leaf_block.shape
            else:
                assert leaf_block.shape == shape
        leaf_block: Block = self.cluster_state.get_block(leafs[0].block_id)
        return leaf_block.size()

    def shape(self):
        for _, leaf in self.leafs_dict.items():
            return leaf.shape()
        for _, tnode in self.children_dict.items():
            return tnode.shape()


class TreeReductionOp(TreeNode):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None, seed=1337):
        super().__init__(cluster_state, tree_node_id)
        self.op_name = None
        # For sampling pairs of leafs in get_actions.
        self.rs = np.random.RandomState(seed)
        self.children_dict: dict = {}
        self.leafs_dict: dict = {}
        # List of actions generated upon first invocation to get_actions.
        # Not a great pattern (side effect), but it's faster than regenerating every time.
        self.action_leaf_q = []

    def __repr__(self):
        return "Reduc(id=%s, op=%s, in=%d)" % (str(self.tree_node_id),
                                               self.op_name,
                                               len(self.children_dict))

    def get_children(self):
        return [self.children_dict[key] for key in sorted(self.children_dict.keys())]

    def num_nodes(self):
        r = 1
        for _, child in self.children_dict.items():
            r += child.num_nodes()
        return r

    def copy(self, cluster_state, parent=None, new_ids=False):
        rop: TreeReductionOp = TreeReductionOp(cluster_state,
                                               None if new_ids else self.tree_node_id)
        assert (rop.tree_node_id is not None
                and (new_ids or rop.tree_node_id == self.tree_node_id))
        rop.parent = parent
        rop.op_name = self.op_name
        rop.copy_on_op = self.copy_on_op
        # This is just a list ids (integers); copy it directly.
        rop.action_leaf_q = copy.deepcopy(self.action_leaf_q)
        for child_id, child in self.children_dict.items():
            child_copy: TreeNode = child.copy(cluster_state=cluster_state,
                                              parent=rop,
                                              new_ids=new_ids)
            assert (child_copy.tree_node_id is not None
                    and (new_ids or child_copy.tree_node_id == child_id))
            rop.children_dict[child_copy.tree_node_id] = child_copy
            if child.tree_node_id in self.leafs_dict:
                rop.leafs_dict[child_copy.tree_node_id] = child_copy
        # TODO (hme): How do we properly copy random state?
        return rop

    def add_child(self, child: TreeNode):
        assert child not in self.children_dict
        self.children_dict[child.tree_node_id] = child
        if isinstance(child, Leaf):
            self.leafs_dict[child.tree_node_id] = child

    def test_integrity(self):
        # This is expensive and only used for testing.
        for leaf_id, leaf in self.leafs_dict.items():
            assert leaf_id == leaf.tree_node_id
        for child_id, child in self.children_dict.items():
            assert child_id == child.tree_node_id
            if isinstance(child, Leaf):
                assert child.tree_node_id in self.leafs_dict

    def update_child(self, old_children, new_children):
        # TODO: Remove integrity checks.
        # self.test_integrity()
        for old_child in old_children:
            assert old_child.tree_node_id in self.children_dict, "Failed to update child: Old " \
                                                                 "child isn't a child of this node."
            del self.children_dict[old_child.tree_node_id]
            if old_child.tree_node_id in self.leafs_dict:
                del self.leafs_dict[old_child.tree_node_id]
        for new_child in new_children:
            self.children_dict[new_child.tree_node_id] = new_child
            if isinstance(new_child, Leaf):
                self.leafs_dict[new_child.tree_node_id] = new_child
        # self.test_integrity()

    def get_leafs(self):
        leafs = []
        for child_id, child in self.children_dict.items():
            leafs += child.get_leafs()
        return leafs

    def is_frontier(self):
        # This is a frontier if all children are computed.
        # This is a stronger constraint than just 2 leafs, but allows
        # for better pairing of operations during action selction.
        return len(self.leafs_dict) == len(self.children_dict)

    def get_frontier(self):
        # This poses an interesting generalization to our prior assumptions about frontiers.
        # We can now have this node be a frontier, as there are actions we can perform on it.
        # It may also contain children that are also frontiers, so collect those.
        # We generate the set of actions from these frontier nodes using their
        # respective actions methods.
        frontier_nodes = []
        if self.is_frontier():
            frontier_nodes.append(self)
        for child_id, child in self.children_dict.items():
            frontier_nodes += child.get_frontier()
        return frontier_nodes

    def _group_leafs(self):
        tree_nodes = sorted(list(self.leafs_dict.values()), key=lambda x: x.tree_node_id)
        grouped_leafs = {}
        leaf_set = set()
        for cnode_id in self.cluster_state.get_cluster_node_ids():
            if cnode_id not in grouped_leafs:
                grouped_leafs[cnode_id] = set()
            for tree_node in tree_nodes:
                assert isinstance(tree_node, Leaf)
                leaf: Leaf = tree_node
                if cnode_id in self.cluster_state.get_block_node_ids(leaf.block_id):
                    if leaf.tree_node_id not in leaf_set:
                        grouped_leafs[cnode_id].add(leaf.tree_node_id)
                        leaf_set.add(leaf.tree_node_id)
        assert len(leaf_set) == len(tree_nodes)
        return grouped_leafs

    def _get_actions(self, leaf_ids, **kwargs):
        assert len(leaf_ids) == 2
        use_all_nodes = kwargs.get("use_all_nodes", False)
        actions = []
        if use_all_nodes:
            node_ids = self.cluster_state.get_cluster_node_ids()
        else:
            # Restrict node ids to the nodes on which the leafs already reside.
            left: Leaf = self.leafs_dict[leaf_ids[0]]
            right: Leaf = self.leafs_dict[leaf_ids[1]]
            node_ids = self.cluster_state.union_nodes(left.block_id, right.block_id)
        for node_id in node_ids:
            actions.append((self.tree_node_id, {"node_id": node_id, "leaf_ids": leaf_ids}))
        return actions

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a cluster node id, second is a pair of leaf_ids.
        """
        if self.is_frontier():
            if len(self.action_leaf_q) == 0:
                if len(self.leafs_dict) == 1:
                    # The ReductionOp should have returned the last leaf upon executing
                    # the last pair of leaves.
                    raise Exception("Unexpected state.")
                grouped_leafs: dict = self._group_leafs()
                for leaf_set in grouped_leafs.values():
                    for tnode_id in leaf_set:
                        self.action_leaf_q.append(tnode_id)
            leaf_id_pair = tuple(self.action_leaf_q[:2])
            return self._get_actions(leaf_id_pair)
        return []

    def simulate_on(self, node_id, leaf_ids=None) -> np.ndarray:
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_op(self._mem_cost(leafs),
                                                   left.block_id,
                                                   right.block_id,
                                                   node_id,
                                                   resources)
        return resources

    def execute_on(self, node_id, leaf_ids=None, plan_only=False) -> TreeNode:
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        result = self._collapse(node_id, left, right, plan_only=plan_only)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]

        # Update action leaf queue.
        assert set(leaf_ids) == {self.action_leaf_q.pop(0), self.action_leaf_q.pop(0)}
        self.action_leaf_q.append(new_leaf.tree_node_id)

        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_op(self._mem_cost(leafs),
                                     left.block_id,
                                     right.block_id,
                                     node_id)
        # Update cluster state with new block.
        self.cluster_state.add_block(new_block, [node_id])
        assert self.cluster_state.blocks_local(left.block_id, right.block_id)
        assert self.cluster_state.blocks_local(left.block_id, new_leaf.block_id)
        # The following are mutating operations.
        # Set the new leaf's parent to this node.
        new_leaf.parent = self
        # Update this node's children: We've collapsed two child leafs by performing
        # the reduction operation, so remove those leafs and replace them with the new leaf.
        self.update_child(leafs, [new_leaf])
        if len(self.children_dict) == 1:
            assert tuple(self.children_dict.values())[0] is new_leaf
            # This was constructed as a reduction with two children,
            # otherwise the reduction would have been transformed into a binary op.
            # We can return the leaf,
            # but we need to perform some mutations to remove this node from the graph.
            # Remove the node from parent reference.
            if self.parent is not None:
                self.parent.update_child([self], [new_leaf])
            # Remove the node as new_leaf's parent.
            new_leaf.parent = self.parent
            return new_leaf
        return self

    def _collapse(self, node_id, left: Leaf, right: Leaf, plan_only):
        lblock: Block = self.cluster_state.get_block(left.block_id)
        rblock: Block = self.cluster_state.get_block(right.block_id)
        if self.op_name == "matmul":
            op_name, args = "tensordot", {"axes": 1}
            assert lblock.shape[1] == rblock.shape[0]
        else:
            op_name, args = self.op_name, {}
            assert lblock.shape == rblock.shape
        options: dict = self.cluster_state.system.get_options(node_id,
                                                              self.cluster_state.cluster_shape)
        block: Block = lblock.bop(op_name, rblock, args=args, options=options, plan_only=plan_only)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block_id = block.id
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self, leafs):
        # Computes the memory required to perform this operation.
        # We approximate by just computing the memory required to store the result.
        assert leafs is not None and len(leafs) > 0
        shape = None
        for leaf in leafs:
            assert leaf.tree_node_id in self.leafs_dict
            leaf_block: Block = self.cluster_state.get_block(leaf.block_id)
            if shape is None:
                shape = leaf_block.shape
            else:
                assert leaf_block.shape == shape
        leaf_block: Block = self.cluster_state.get_block(leafs[0].block_id)
        return leaf_block.size()

    def shape(self):
        for _, leaf in self.leafs_dict.items():
            return leaf.shape()
        for _, tnode in self.children_dict.items():
            return tnode.shape()
