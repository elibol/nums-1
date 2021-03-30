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
import scipy.special

from nums.core.storage.storage import ArrayGrid
from nums.core.optimizer.cluster_sim import ClusterState
from nums.core.array.base import BlockArrayBase, Block
from nums.core.systems.systems import System
from nums.core.array import utils as array_utils


def subsample(total_items, max_items, rs: np.random.RandomState):
    perms = rs.permutation(total_items)
    if total_items < max_items:
        return perms
    return perms[:max_items]


class Counter(object):

    def __init__(self):
        self.n = -1

    def __call__(self):
        self.n += 1
        return self.n

    def copy(self):
        c_copy = Counter()
        c_copy.n = self.n
        return c_copy


class TreeNode(object):

    new_id = Counter()

    def __init__(self, tree_node_id=None):
        # A deterministic identifier that's preserved across copies.
        # label each node as grid_entry, i, where i \in 0, ..., num nodes,
        # incremented top-down and left-to-right.
        # A collapsed node is a tuple of its id and the child nodes it comprised.
        self.tree_node_id = self.new_id() if tree_node_id is None else tree_node_id
        self.parent: TreeNode = None
        self.cluster_state: ClusterState = None
        self.copy_on_op = True

    def get_root(self):
        if self.parent is None:
            return self
        return self.parent.get_root()

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

    def execute_on(self, node_id, leaf_ids=None):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)

    def make_bop(self, op_name, other, args=None):
        assert isinstance(other, TreeNode)
        bop: BinaryOp = BinaryOp()
        bop.cluster_state = self.cluster_state
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

    def __init__(self, tree_node_id=None):
        # The leaf abstraction enables the same block to be a part of multiple computations,
        # evolving its state across all leafs holding a reference to the block.
        super().__init__(tree_node_id)
        self.block_id = None

    def __repr__(self):
        return str(self.block_id)

    def num_nodes(self):
        return 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        leaf: Leaf = Leaf(None if new_ids else self.tree_node_id)
        assert (leaf.tree_node_id is not None
                and (new_ids or leaf.tree_node_id == self.tree_node_id))
        leaf.cluster_state = cluster_state
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

    def __init__(self, tree_node_id=None):
        super().__init__(tree_node_id)
        self.child: TreeNode = None
        self.op_name = None

    def copy(self, cluster_state, parent=None, new_ids=False):
        uop: UnaryOp = UnaryOp(None if new_ids else self.tree_node_id)
        assert (uop.tree_node_id is not None
                and (new_ids or uop.tree_node_id == self.tree_node_id))
        uop.cluster_state = cluster_state
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

    def execute_on(self, node_id, leaf_ids=None) -> Leaf:
        assert leaf_ids is None
        assert isinstance(self.child, Leaf)
        result = self._collapse(node_id)
        new_leaf: Leaf = result[0]
        new_leaf.cluster_state = self.cluster_state
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

    def _collapse(self, node_id):
        assert isinstance(self.child, Leaf)
        block: Block = self.cluster_state.get_block(self.child.block_id)
        op_name, args = self.op_name,  {}
        options: dict = self.cluster_state.system.get_options(node_id,
                                                              self.cluster_state.cluster_shape)
        if op_name == "transpose":
            block: Block = block.transpose()
        else:
            block: Block = block.ufunc(op_name, options=options)
        leaf: Leaf = Leaf()
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

    def __init__(self, tree_node_id=None):
        super().__init__(tree_node_id)
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
        return "(%s %s %s)" % (str(self.left), bop_symbol, str(self.right))

    def num_nodes(self):
        return self.left.num_nodes() + self.right.num_nodes() + 1

    def copy(self, cluster_state, parent=None, new_ids=False):
        bop = BinaryOp(None if new_ids else self.tree_node_id)
        assert (bop.tree_node_id is not None
                and (new_ids or bop.tree_node_id == self.tree_node_id))
        bop.cluster_state = cluster_state
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

    def execute_on(self, node_id, leaf_ids=None) -> Leaf:
        """
        Update cluster state to reflect the cluster's load after computing this node.
        We generate a leaf node for BinaryOp, updating the leaf node's computation
        time based on object transfer costs, etc.
        """
        assert leaf_ids is None
        assert isinstance(self.left, Leaf) and isinstance(self.right, Leaf)
        result = self._collapse(node_id)
        new_leaf: Leaf = result[0]
        new_leaf.cluster_state = self.cluster_state
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

    def _collapse(self, node_id):
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
        block: Block = lblock.bop(op_name, rblock, args=args, options=options)
        leaf: Leaf = Leaf()
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


class ReductionOp(TreeNode):

    def __init__(self, tree_node_id=None, seed=1337):
        super().__init__(tree_node_id)
        self.op_name = None
        # For sampling pairs of leafs in get_actions.
        self.rs = np.random.RandomState(seed)
        self.children_dict: dict = {}
        self.leafs_dict: dict = {}

    def __repr__(self):
        return self.op_name+"(%d)" % len(self.children_dict)

    def num_nodes(self):
        r = 1
        for _, child in self.children_dict.items():
            r += child.num_nodes()
        return r

    def copy(self, cluster_state, parent=None, new_ids=False):
        rop: ReductionOp = ReductionOp(None if new_ids else self.tree_node_id)
        assert (rop.tree_node_id is not None
                and (new_ids or rop.tree_node_id == self.tree_node_id))
        rop.cluster_state = cluster_state
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
            if self.tree_node_id in self.leafs_dict:
                rop.leafs_dict[child_copy.tree_node_id] = child_copy
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
                elif max_pairs <= num_pairs//2:
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

    def execute_on(self, node_id, leaf_ids=None) -> TreeNode:
        """
        The can return:
        - Another ReductionOp.
        - A BinaryOp.
        """
        assert len(leaf_ids) == 2
        leafs = self.leafs_dict[leaf_ids[0]], self.leafs_dict[leaf_ids[1]]
        left, right = leafs
        assert isinstance(left, Leaf) and isinstance(right, Leaf)
        result = self._collapse(node_id, left, right)
        new_leaf: Leaf = result[0]
        new_leaf.cluster_state = self.cluster_state
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
        if len(self.children_dict) > 2:
            # Return self if there are more than 2 children left.
            # We need to perform further reductions before this node can be converted to a BinaryOp.
            return self
        elif len(self.children_dict) == 2:
            bop: BinaryOp = BinaryOp()
            bop.cluster_state = self.cluster_state
            bop.op_name = self.op_name
            bop.copy_on_op = self.copy_on_op
            # These need not be leaf nodes.
            # This is a reduction op => commutative => we don't care about order of ops.
            bop.left, bop.right = tuple(self.children_dict.values())
            # We're transforming this node, so update child references.
            bop.left.parent = bop
            bop.right.parent = bop
            # Update parent references.
            if self.parent is not None:
                self.parent.update_child([self], [bop])
            bop.parent = self.parent
            return bop
        elif len(self.children_dict) == 1:
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
            raise Exception("Unexpected number of children %d" % len(self.children_dict))

    def _collapse(self, node_id, left: Leaf, right: Leaf):
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
        block: Block = lblock.bop(op_name, rblock, args=args, options=options)
        leaf: Leaf = Leaf()
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


class GraphArray(object):

    @staticmethod
    def graphs_from_ba(ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op) -> np.ndarray:
        graphs = np.empty(shape=ba.grid.grid_shape, dtype=np.object)
        for grid_entry in ba.grid.get_entry_iterator():
            block: Block = ba.blocks[grid_entry]
            # Allocate the block to the node on which it's created.
            node_id = cluster_state.get_cluster_entry(block.true_grid_entry(), ba.grid.grid_shape)
            cluster_state.add_block(block, node_ids=[node_id])
            cluster_state.init_mem_load(node_id, block.id)

            # Create the leaf representing this block for future computations.
            leaf: Leaf = Leaf()
            leaf.cluster_state = cluster_state
            leaf.block_id = block.id
            leaf.copy_on_op = copy_on_op
            graphs[grid_entry] = leaf
        return graphs

    @classmethod
    def from_ba(cls, ba: BlockArrayBase, cluster_state: ClusterState, copy_on_op=True):
        return GraphArray(ba.grid, cluster_state, GraphArray.graphs_from_ba(ba,
                                                                            cluster_state,
                                                                            copy_on_op),
                          copy_on_op=copy_on_op)

    def __init__(self, grid: ArrayGrid, cluster_state: ClusterState, graphs: np.ndarray,
                 copy_on_op=True):
        self.grid = grid
        self.cluster_state = cluster_state
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.dtype = self.grid.dtype
        self.graphs = graphs
        self.copy_on_op = copy_on_op

    def __repr__(self):
        return str(self.graphs)

    def copy(self, new_ids=False):
        new_cluster = self.cluster_state.copy()
        graphs_copy = np.empty(shape=self.grid.grid_shape, dtype=np.object)
        for grid_entry in self.grid.get_entry_iterator():
            old_tree_node: TreeNode = self.graphs[grid_entry]
            # The recursive copy should go through without issue,
            # since nodes only hold reference to cluster_state and block ids.
            graphs_copy[grid_entry] = old_tree_node.copy(cluster_state=new_cluster,
                                                         new_ids=new_ids)
        return GraphArray(self.grid, new_cluster, graphs_copy)

    def to_blocks(self) -> np.ndarray:
        blocks: np.ndarray = np.empty(self.grid.grid_shape, dtype=Block)
        for grid_entry in self.grid.get_entry_iterator():
            leaf: TreeNode = self.graphs[grid_entry]
            assert isinstance(leaf, Leaf), "%s,%s" % (str(leaf), type(leaf))
            blocks[grid_entry] = self.cluster_state.get_block(leaf.block_id)
        return blocks

    def other_to_ba(self, other):
        if isinstance(other, GraphArray):
            return other
        return self.from_ba(other, self.cluster_state)

    def tensordot(self, other, axes=2):
        other = self.other_to_ba(other)
        # TODO: Reuse BlockArrayBase tensordot operator.
        this_axes = self.grid.grid_shape[:-axes]
        this_sum_axes = self.grid.grid_shape[-axes:]
        other_axes = other.grid.grid_shape[axes:]
        other_sum_axes = other.grid.grid_shape[:axes]
        assert this_sum_axes == other_sum_axes
        result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
        result_block_shape = tuple(self.block_shape[:-axes] + other.block_shape[axes:])
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=self.dtype.__name__)
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                # A \in \R^{I \times K}
                # B \in \R^{K \times J}
                # C \in \R^{I \times J}
                # C[i, j] = sum_k^K { A[i, k] * B[k, j] }
                grid_entry = tuple(i + j)
                if len(sum_dims) == 1:
                    k = sum_dims[0]
                    self_node: TreeNode = self.graphs[tuple(i + k)]
                    other_node: TreeNode = other.graphs[tuple(k + j)]
                    dot_node: TreeNode = self_node.tensordot(other_node, axes=axes)
                    result_graphs[grid_entry] = dot_node
                else:
                    add_reduce_op = ReductionOp()
                    add_reduce_op.cluster_state = self.cluster_state
                    add_reduce_op.op_name = "add"
                    add_reduce_op.copy_on_op = self.copy_on_op
                    for k in sum_dims:
                        self_node: TreeNode = self.graphs[tuple(i + k)]
                        other_node: TreeNode = other.graphs[tuple(k + j)]
                        dot_node: TreeNode = self_node.tensordot(other_node, axes=axes)
                        # Explicitly add parent here, since sum depends on prod.
                        # Not needed for other ops; make_bop takes care of it.
                        # We don't need to copy the node here since the local
                        # tree structure here is never exposed.
                        dot_node.parent = add_reduce_op
                        add_reduce_op.add_child(dot_node)
                    result_graphs[grid_entry] = add_reduce_op
        return GraphArray(result_grid, self.cluster_state, result_graphs,
                          copy_on_op=self.copy_on_op)

    def __matmul__(self, other):
        return self.tensordot(other, axes=1)

    def ga_from_arr(self, arr, result_shape):
        sample_idx = tuple(0 for dim in arr.shape)
        if isinstance(arr, TreeNode):
            sample_node: TreeNode = arr
            assert result_shape == ()
        else:
            sample_node: TreeNode = arr[sample_idx]
        result_block_shape = sample_node.shape()
        result_dtype_str = self.grid.dtype.__name__
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=result_dtype_str)
        assert arr.shape == result_grid.grid_shape
        return GraphArray(result_grid, self.cluster_state, arr, copy_on_op=self.copy_on_op)

    def __add__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(self.graphs + other.graphs,
                                array_utils.broadcast_shape(self.shape, other.shape))

    def __sub__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(self.graphs - other.graphs,
                                array_utils.broadcast_shape(self.shape, other.shape))

    def __mul__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(self.graphs * other.graphs,
                                array_utils.broadcast_shape(self.shape, other.shape))

    def __truediv__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(self.graphs / other.graphs,
                                array_utils.broadcast_shape(self.shape, other.shape))

    def __pow__(self, other):
        other = self.other_to_ba(other)
        return self.ga_from_arr(self.graphs ** other.graphs,
                                array_utils.broadcast_shape(self.shape, other.shape))

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __imatmul__ = __matmul__
    __itruediv__ = __truediv__
    __ipow__ = __pow__

    __radd__ = __add__

    def __rsub__(self, other):
        other = self.other_to_ba(other)
        return other - self

    __rmul__ = __mul__

    def __rmatmul__(self, other):
        other = self.other_to_ba(other)
        return other @ self

    def __rtruediv__(self, other):
        other = self.other_to_ba(other)
        return other / self

    def __rpow__(self, other):
        other = self.other_to_ba(other)
        return other ** self

    # Unary operators.
    def __neg__(self):
        return self.ufunc("negative")

    def __pos__(self):
        return self.ufunc("positive")

    def ufunc(self, op_name):
        result_grid = self.grid.copy()
        result_graphs = np.empty(shape=result_grid.grid_shape, dtype=np.object)
        for grid_entry in self.grid.get_entry_iterator():
            self._add_uop(op_name, grid_entry, self.graphs, result_graphs)
        return GraphArray(result_grid, self.cluster_state, result_graphs,
                          copy_on_op=self.copy_on_op)

    def __getattr__(self, item):
        if item != "T":
            raise NotImplementedError(item)
        metaT = self.grid.to_meta()
        metaT["shape"] = tuple(reversed(metaT["shape"]))
        metaT["block_shape"] = tuple(reversed(metaT["block_shape"]))
        result_grid: ArrayGrid = ArrayGrid.from_meta(metaT)
        result_graphs = np.copy(self.graphs.T)
        for grid_entry in result_grid.get_entry_iterator():
            self._add_uop("transpose", grid_entry, result_graphs, result_graphs)
        return GraphArray(result_grid, self.cluster_state, result_graphs,
                          copy_on_op=self.copy_on_op)

    def _add_uop(self, op_name, grid_entry, old_arr, new_arr):
        uop: UnaryOp = UnaryOp()
        uop.cluster_state = self.cluster_state
        uop.copy_on_op = self.copy_on_op
        uop.op_name = op_name
        # Do this in case old_arr == new_arr.
        old_root: TreeNode = old_arr[grid_entry]
        assert old_root.parent is None
        if self.copy_on_op:
            # Need to copy here, in case old_root is used in other operations.
            # We could eliminate this requirement by maintaining multiple parents,
            # but this breaks a lot of assumptions.
            uop.child = old_root.copy(uop.cluster_state,
                                      parent=uop,
                                      new_ids=True)
        else:
            assert old_root.parent is None
            uop.child = old_root
            old_root.parent = uop
        new_arr[grid_entry] = uop
