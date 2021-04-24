import ray
import numpy as np

import nums.numpy as nps
from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

# settings.cluster_shape = (...)
# ray.init(address="...")


# System API Custom Scheduling
app: ArrayApplication = instance()
system: System = app.system
A: BlockArray = app.random.normal(shape=(6, 6), block_shape=(3, 3))
B: BlockArray = app.random.normal(shape=(6, 6), block_shape=(3, 3))

grid: ArrayGrid = ArrayGrid(A.shape, A.block_shape, dtype="float64")
C: BlockArray = BlockArray(grid, app.system)

for grid_entry in grid.get_entry_iterator():
    A_block: Block = A.blocks[grid_entry]
    B_block: Block = B.blocks[grid_entry]
    C_block: Block = C.blocks[grid_entry]
    cluster_entry = tuple(np.array(grid_entry) % np.array(settings.cluster_shape))
    print(cluster_entry)
    options = system.get_options(cluster_entry, settings.cluster_shape)
    C_block.oid = system.call_with_options("add",
                                           (A_block.oid, B_block.oid),
                                           {},
                                           options=options)

print(C.get())


# System API BlockCyclic Scheduling
# app: ArrayApplication = instance()
# system: System = app.system
# A: BlockArray = app.random.normal(shape=(6, 6), block_shape=(3, 3))
# B: BlockArray = app.random.normal(shape=(6, 6), block_shape=(3, 3))
#
# grid: ArrayGrid = ArrayGrid(A.shape, A.block_shape, dtype="float64")
# C: BlockArray = BlockArray(grid, app.system)
#
# for grid_entry in grid.get_entry_iterator():
#     A_block: Block = A.blocks[grid_entry]
#     B_block: Block = B.blocks[grid_entry]
#     C_block: Block = C.blocks[grid_entry]
#     C_block.oid = system.add(A_block.oid, B_block.oid, syskwargs={
#         "grid_entry": grid_entry,
#         "grid_shape": grid.grid_shape
#     })
#     print(grid_entry)
#
# print(C.get())

# Array App API

# app: ArrayApplication = instance()
#
# A = app.random.normal(shape=(4, 4), block_shape=(2, 2))
# B = app.random.normal(shape=(4, 4), block_shape=(2, 2))
#
# C = A + B
#
# print(C.get())

# NumPy API

# A: BlockArray = nps.random.randn(*(10**8, 2))
# B: BlockArray = nps.random.randn(*(10**8, 2))
#
# print(A.shape, A.block_shape, A.grid.grid_shape)
# print(B.shape, B.block_shape, B.grid.grid_shape)
#
# C: BlockArray = A + B
#
# print(C.shape, C.block_shape, C.grid.grid_shape)

# shape <= block_shape * grid_shape
