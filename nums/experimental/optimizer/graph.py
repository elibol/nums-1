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


def subsample(total_items, max_items, rs: np.random.RandomState):
    perms = rs.permutation(total_items)
    if total_items < max_items:
        return perms
    return perms[:max_items]


class TreeNode(object):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        # A deterministic identifier that's preserved across copies.
        # label each node as grid_entry, i, where i \in 0, ..., num nodes,
        # incremented top-down and left-to-right.
        self.cluster_state: ClusterState = cluster_state
        self.tree_node_id = self.cluster_state.counter() if tree_node_id is None else tree_node_id
        self.parent: TreeNode = None
        self.copy_on_op = True

    def get_root(self):
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_children(self):
        raise NotImplementedError()

    def num_nodes(self):
        raise NotImplementedError()

    def copy(self, cluster_state, parent=None, new_ids=False):
        raise NotImplementedError()

    def update_child(self, old_children, new_children):
        raise NotImplementedError()

    def get_leafs(self):
        raise NotImplementedError()

    def is_frontier(self):
        raise NotImplementedError()

    def get_frontier(self):
        raise NotImplementedError()

    def get_actions(self, **kwargs):
        raise NotImplementedError()

    def simulate_on(self, node_id, leaf_ids=None) -> np.ndarray:
        raise NotImplementedError()

    def execute_on(self, node_id, leaf_ids=None, plan_only=False):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)

    def make_bop(self, op_name, other, args=None):
        assert isinstance(other, TreeNode)
        bop: BinaryOp = BinaryOp(self.cluster_state)
        bop.op_name = op_name
        bop.args = args
        assert self.copy_on_op == other.copy_on_op
        bop.copy_on_op = self.copy_on_op
        # Need to copy here in case self and other are used in other operations.
        if self.copy_on_op:
            bop.left = self.copy(bop.cluster_state, parent=bop, new_ids=True)
            bop.right = other.copy(bop.cluster_state, parent=bop, new_ids=True)
        else:
            assert self.parent is None and other.parent is None
            bop.left, bop.right = self, other
            bop.left.parent, bop.right.parent = bop, bop
        return bop

    def tensordot(self, other, axes):
        return self.make_bop("tensordot", other, args={"axes": axes})

    def __matmul__(self, other):
        return self.make_bop("matmul", other)

    def __add__(self, other):
        return self.make_bop("add", other)

    def __sub__(self, other):
        return self.make_bop("sub", other)

    def __mul__(self, other):
        return self.make_bop("mul", other)

    def __truediv__(self, other):
        return self.make_bop("truediv", other)

    def __pow__(self, other):
        return self.make_bop("pow", other)


class Leaf(TreeNode):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        # The leaf abstraction enables the same block to be a part of multiple computations,
        # evolving its state across all leafs holding a reference to the block.
        super().__init__(cluster_state, tree_node_id)
        self.block_id = None

    def get_children(self):
        return []

    def __repr__(self):
        return "Leaf(id=%s, bid=%s)" % (str(self.tree_node_id), str(self.block_id))

    def num_nodes(self):
        return 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        leaf: Leaf = Leaf(cluster_state, None if new_ids else self.tree_node_id)
        assert (leaf.tree_node_id is not None
                and (new_ids or leaf.tree_node_id == self.tree_node_id))
        leaf.parent = parent
        leaf.block_id = self.block_id
        leaf.copy_on_op = self.copy_on_op
        return leaf

    def get_leafs(self):
        return [self]

    def is_frontier(self):
        return False

    def get_frontier(self):
        return []

    def get_actions(self, **kwargs):
        return []

    def shape(self):
        return self.cluster_state.get_block(self.block_id).shape


class UnaryOp(TreeNode):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.child: TreeNode = None
        self.op_name = None

    def get_children(self):
        return [self.child]

    def copy(self, cluster_state, parent=None, new_ids=False):
        uop: UnaryOp = UnaryOp(cluster_state, None if new_ids else self.tree_node_id)
        assert (uop.tree_node_id is not None
                and (new_ids or uop.tree_node_id == self.tree_node_id))
        uop.parent = parent
        uop.child = self.child.copy(cluster_state, parent=uop, new_ids=new_ids)
        uop.op_name = self.op_name
        uop.copy_on_op = self.copy_on_op
        return uop

    def update_child(self, old_children, new_children):
        assert len(old_children) == len(new_children) == 1
        old_child, new_child = old_children[0], new_children[0]
        self.child = new_child

    def get_leafs(self):
        return self.child.get_leafs()

    def is_frontier(self):
        return isinstance(self.child, Leaf)

    def get_frontier(self):
        if self.is_frontier():
            return [self]
        else:
            return self.child.get_frontier()

    def num_nodes(self):
        return self.child.num_nodes() + 1

    def get_actions(self, **kwargs):
        actions = []
        if self.is_frontier():
            use_all_nodes = kwargs.get("use_all_nodes", False)
            if use_all_nodes:
                node_ids = self.cluster_state.get_cluster_node_ids()
            else:
                # Restrict node ids to the nodes on which the leafs already reside.
                node_ids = self.cluster_state.get_block_node_ids(self.child.block_id)
            for node_id in node_ids:
                actions.append((self.tree_node_id, {"node_id": node_id}))
        return actions

    def simulate_on(self, node_id, leaf_ids=None) -> np.ndarray:
        assert leaf_ids is None
        assert isinstance(self.child, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_uop(self._mem_cost(),
                                                    self.child.block_id,
                                                    node_id,
                                                    resources)
        return resources

    def execute_on(self, node_id, leaf_ids=None, plan_only=False) -> Leaf:
        assert leaf_ids is None
        assert isinstance(self.child, Leaf)
        result = self._collapse(node_id, plan_only)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        self.cluster_state.commit_uop(self._mem_cost(),
                                      self.child.block_id,
                                      node_id)
        self.cluster_state.add_block(new_block, [node_id])
        assert self.cluster_state.blocks_local(self.child.block_id, new_leaf.block_id)
        new_leaf.parent = self.parent
        if self.parent is not None:
            self.parent.update_child([self], [new_leaf])
        return new_leaf

    def _collapse(self, node_id, plan_only):
        assert isinstance(self.child, Leaf)
        block: Block = self.cluster_state.get_block(self.child.block_id)
        op_name, args = self.op_name,  {}
        options: dict = self.cluster_state.system.get_options(node_id,
                                                              self.cluster_state.cluster_shape)
        if op_name == "transpose":
            block: Block = block.transpose(plan_only=plan_only)
        else:
            block: Block = block.ufunc(op_name, options=options, plan_only=plan_only)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block_id = block.id
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _mem_cost(self):
        assert isinstance(self.child, Leaf)
        block: Block = self.cluster_state.get_block(self.child.block_id)
        return np.product(block.shape)

    def shape(self):
        child_shape = self.child.shape()
        if self.op_name == "transpose":
            return tuple(reversed(child_shape))
        else:
            return child_shape


class BinaryOp(TreeNode):

    def __init__(self, cluster_state: ClusterState, tree_node_id=None):
        super().__init__(cluster_state, tree_node_id)
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.op_name = None
        self.args = None

    def __repr__(self):
        bop_symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "truediv": "/",
            "matmul": "@",
            "tensordot": "@"
        }[self.op_name]
        return "BOp(id=%s, op=%s%s%s)" % (self.tree_node_id, str(self.left.tree_node_id),
                                             bop_symbol, str(self.right.tree_node_id))

    def get_children(self):
        return [self.left, self.right]

    def num_nodes(self):
        return self.left.num_nodes() + self.right.num_nodes() + 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        bop = BinaryOp(cluster_state, None if new_ids else self.tree_node_id)
        assert (bop.tree_node_id is not None
                and (new_ids or bop.tree_node_id == self.tree_node_id))
        bop.parent = parent
        bop.op_name = self.op_name
        bop.args = None if self.args is None else self.args.copy()
        bop.left = self.left.copy(cluster_state, bop, new_ids=new_ids)
        bop.right = self.right.copy(cluster_state, bop, new_ids=new_ids)
        bop.copy_on_op = self.copy_on_op
        return bop

    def update_child(self, old_children, new_children):
        assert len(old_children) == len(new_children) == 1
        old_child, new_child = old_children[0], new_children[0]
        if old_child == self.left:
            self.left = new_child
        elif old_child == self.right:
            self.right = new_child
        else:
            raise Exception("Failed to update child: Old child doesn't this nodes children.")

    def get_leafs(self):
        return self.left.get_leafs() + self.right.get_leafs()

    def is_frontier(self):
        return isinstance(self.left, Leaf) and isinstance(self.right, Leaf)

    def get_frontier(self):
        if self.is_frontier():
            # This is a frontier node.
            return [self]
        return self.left.get_frontier() + self.right.get_frontier()

    def get_actions(self, **kwargs):
        """
        Returns a list of actions.
        An action is a tuple: First entry is a function. Second entry is kwargs.
        Invoked actions return a new node without mutating the tree,
        which is always a leaf for BinaryOp.
        """
        actions = []
        if self.is_frontier():
            use_all_nodes = kwargs.get("use_all_nodes", False)
            if use_all_nodes:
                node_ids = self.cluster_state.get_cluster_node_ids()
            else:
                # Restrict node ids to the nodes on which the leafs already reside.
                node_ids = self.cluster_state.union_nodes(self.left.block_id, self.right.block_id)
            for node_id in node_ids:
                actions.append((self.tree_node_id, {"node_id": node_id}))
        return actions

    def simulate_on(self, node_id, leaf_ids=None) -> np.ndarray:
        assert leaf_ids is None
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        resources = self.cluster_state.resources.copy()
        resources = self.cluster_state.simulate_op(self._mem_cost(),
                                                   self.left.block_id,
                                                   self.right.block_id,
                                                   node_id,
                                                   resources)
        return resources

    def execute_on(self, node_id, leaf_ids=None, plan_only=False) -> Leaf:
        """
        Update cluster state to reflect the cluster's load after computing this node.
        We generate a leaf node for BinaryOp, updating the leaf node's computation
        time based on object transfer costs, etc.
        """
        assert leaf_ids is None
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        result = self._collapse(node_id, plan_only)
        new_leaf: Leaf = result[0]
        new_block: Block = result[1]
        # This updates load on nodes and channels.
        # This also updates block states to indicate that they now reside on the provided nodes.
        # Update the cluster state after computing the leaf, so that transfer costs are properly
        # captured by leaf node computations.
        self.cluster_state.commit_op(self._mem_cost(),
                                     self.left.block_id,
                                     self.right.block_id,
                                     node_id)
        # Update cluster state with new block.
        self.cluster_state.add_block(new_block, [node_id])
        assert self.cluster_state.blocks_local(self.left.block_id, self.right.block_id)
        assert self.cluster_state.blocks_local(self.left.block_id, new_leaf.block_id)
        # These are mutating operations.
        # Eliminate references to this node and replace them with leaf.
        new_leaf.parent = self.parent
        if self.parent is not None:
            self.parent.update_child([self], [new_leaf])
        return new_leaf

    def _collapse(self, node_id, plan_only):
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        lblock: Block = self.cluster_state.get_block(self.left.block_id)
        rblock: Block = self.cluster_state.get_block(self.right.block_id)
        if self.op_name == "matmul":
            op_name, args = "tensordot",  {"axes": 1}
        elif self.op_name == "tensordot":
            op_name, args = "tensordot", self.args
        else:
            op_name, args = self.op_name,  {}
            assert array_utils.can_broadcast_shapes(lblock.shape, rblock.shape)
        options: dict = self.cluster_state.system.get_options(node_id,
                                                              self.cluster_state.cluster_shape)
        block: Block = lblock.bop(op_name, rblock, args=args, options=options, plan_only=plan_only)
        leaf: Leaf = Leaf(self.cluster_state)
        leaf.block_id = block.id
        leaf.copy_on_op = self.copy_on_op
        return leaf, block

    def _tdop_shape(self, left_shape, right_shape):
        assert isinstance(self.args, dict)
        axes = self.args.get("axes", 1)
        this_sum_axes = left_shape[-axes:]
        other_sum_axes = right_shape[:axes]
        assert this_sum_axes == other_sum_axes
        return tuple(left_shape[:-axes] + right_shape[axes:])

    def _mem_cost(self):
        # Computes the memory required to perform this operation.
        # We approximate by just computing the memory required to store the result.
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        lblock: Block = self.cluster_state.get_block(self.left.block_id)
        rblock: Block = self.cluster_state.get_block(self.right.block_id)
        if self.op_name == "matmul" or self.op_name == "tensordot":
            output_shape = self._tdop_shape(lblock.shape, rblock.shape)
        else:
            assert array_utils.can_broadcast_shapes(lblock.shape, rblock.shape)
            output_shape = array_utils.broadcast_shape(lblock.shape, rblock.shape)
        return np.product(output_shape)

    def shape(self):
        left_shape = self.left.shape()
        right_shape = self.right.shape()
        if self.op_name == "matmul" or self.op_name == "tensordot":
            return self._tdop_shape(left_shape, right_shape)
        else:
            return array_utils.broadcast_shape(left_shape, right_shape)
