import sys
import functools
import time
import itertools
import gc
from collections import defaultdict
from typing import Tuple
import numpy as np
import ray

from nums.core.systems import numpy_compute
from nums.core.settings import np_ufunc_map
from nums.core.systems.interfaces import RNGInterface
from nums.core.systems.utils import extract_functions


def cupy_used_bytes():
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes()


class BaseGPUSystem(object):
    def __init__(self):
        for name in ['random_block', 'new_block', 'update_block', 'create_block',
            'sum_reduce', 'map_uop', 'reshape', 'inv', 'empty', 'reduce_axis',
            'astype', 'bop']:
            setattr(self, name, functools.partial(self.call_compute_interface, name))

    def get_rng(self, seed) -> RNGInterface:
        from nums.core.systems import numpy_compute
        self.rng_cls = numpy_compute.RNG
        return self.rng_cls(seed)

    def init(self):
        pass

    def shutdown(self):
        pass

    def register(self, name: str, func: callable, remote_params: dict = None):
        pass

    def call_compute_interface(self, name, *args, **kwargs):
        raise NotImplementedError


##############################################################
############ SerialSystem: Serial implementation #############
##############################################################
class SerialSystem(BaseGPUSystem):
    def __init__(self, compute_module):
        # Init ComputeInterface
        self.compute_imp = compute_module.ComputeCls()
        super().__init__()

    def call_compute_interface(self, name, *args, **kwargs):
        del kwargs['syskwargs']
        #if name in ['bop', 'map_uop']:
        #    print(f"SerialSystem::call compute {name} {args[0]}")
        #else:
        #    print(f"SerialSystem::call compute {name}")
        ret =  getattr(self.compute_imp, name)(*args, **kwargs)
        #print(f"SerialSystem::result {ret.shape} {cupy_used_bytes()/1e9} {ret.dtype}")
        return ret


class NumpySerialSystem(SerialSystem):
    def __init__(self, num_gpus):
        super().__init__(numpy_compute)

    def put(self, x):
        return x

    def get(self, x):
        return x

    def touch(self, object_id, syskwargs):
        return object_id


class CupySerialSystem(SerialSystem):
    def __init__(self, num_gpus):
        import cupy as cp
        from nums.core.systems import cupy_compute

        self.cp = cp
        super().__init__(cupy_compute)

    def put(self, x):
        return self.cp.array(x)

    def get(self, x):
        self.cp.cuda.Device(0).synchronize()
        if isinstance(x, list):
            return [a.get() for a in x]
        else:
            return x.get()

    def touch(self, object_id, syskwargs):
        self.cp.cuda.Device(0).synchronize()
        return object_id

    def shutdown(self):
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()

##############################################################
########## ParallelSystem: Parallel implementation ###########
##############################################################
class CupySystemArrRef:
    def __init__(self, cp_arr, system):
        self.cp_arr = cp_arr
        self.system = system

    def __del__(self):
        self.system.delete(self.cp_arr)


class CupyParallelSystem(BaseGPUSystem):
    def __init__(self, local_cache=True, immediate_gc=False):
        import cupy as cp
        from nums.core.systems import cupy_compute

        self.cp = cp
        self.num_gpus = 1
        self.local_cache = local_cache
        self.immediate_gc = immediate_gc
        self.cluster_shape = (self.num_gpus, 1)
        self.optimizer = True
        self.compute_imp = cupy_compute.ComputeCls()
        self.dist_dict = defaultdict(dict)   # Dict[hash(array) -> Dict[actor_id -> array]]
        super().__init__()

    def put(self, x):
        with self.cp.cuda.Device(0):
            ret = self.cp.array(x)
        ret = self._register_new_array(ret, 0)

        for actor_id in range(1, self.num_gpus):
            self._distribute_to(ret, actor_id)

        return CupySystemArrRef(ret, self)

    def get(self, x):
        if isinstance(x, list):
            return [a.cp_arr.get() for a in x]
        else:
            return x.cp_arr.get()

    def touch(self, x, syskwargs):
        x.cp_arr.device.synchronize()
        return x

    def get_cluster_entry(self, grid_entry, grid_shape):
        ret = [0]
        for i in range(len(grid_entry)):
            dim = 1 if i == len(grid_entry) - 1 else grid_shape[i+1]
            ret[0] = (ret[0] + grid_entry[i]) * dim
        ret[0] = ret[0] % self.num_gpus
        ret.append(0)
        return tuple(ret)

    def call_with_options(self, name, args, kwargs, options):
        dst_actor = options["dst_actor"]
        # print(f"CupyParallelSystem::call compute {args} on {dst_actor}")

        args = [self._distribute_to(v.cp_arr, dst_actor)
                if isinstance(v, CupySystemArrRef) else v for v in args]
        kwargs = {k: self._distribute_to(v.cp_arr, dst_actor)
                if isinstance(v, CupySystemArrRef) else v for k, v in kwargs.items()}

        with self.cp.cuda.Device(dst_actor):
            # print(f"CupyParallelSystem::call args {args} kwargs {kwargs}")
            ret = getattr(self.compute_imp, name)(*args, **kwargs)

        if self.immediate_gc:
            self.dist_dict = defaultdict(dict)
        else:
            ret = self._register_new_array(ret, dst_actor)
        return CupySystemArrRef(ret, self)

    def call_compute_interface(self, name, *args, **kwargs):
        # Make device placement decisions
        syskwargs = kwargs.pop('syskwargs')
        grid_entry = syskwargs["grid_entry"]
        grid_shape = syskwargs["grid_shape"]

        if self.optimizer:
            cluster_entry: tuple = self.get_cluster_entry(grid_entry, grid_shape)
            # print(f"CupyParallelSystem::call grid entry {grid_entry} and cluster entry {cluster_entry}")
            dst_actor = cluster_entry[0]
        else:
            if name == 'bop':
                dst_actor = None
                for arg in itertools.chain(args, kwargs.values()):
                    if isinstance(arg, CupySystemArrRef):
                        dst_actor = arg.cp_arr.data.device_id
                        break
            else:
                gid = get_flatten_id(grid_entry, grid_shape)
                dst_actor = gid % self.num_gpus

        options = {}
        options["dst_actor"] = dst_actor
        return self.call_with_options(name, args, kwargs, options)

    def distribute_to(self, arr_ref, dst_actor):
        return self._distribute_to(arr_ref.cp_arr, dst_actor)

    def _distribute_to(self, arr, dst_actor):
        if self.local_cache:
            arr_hash = self._get_array_hash(arr)
            ret = self.dist_dict[arr_hash].get(dst_actor, None)
            if ret is None:
                if arr.data.device_id == dst_actor:
                    ret = arr
                else:
                    # print(f"Copy {arr.shape} from {arr.data.device_id} to {dst_actor}")
                    with self.cp.cuda.Device(dst_actor):
                        ret = self.cp.asarray(arr)
                    self.dist_dict[arr_hash][dst_actor] = ret
        else:
            if arr.data.device_id == dst_actor:
                ret = arr
            else:
                with self.cp.cuda.Device(dst_actor):
                    ret = self.cp.asarray(arr)

        return ret

    def _get_array_hash(self, arr):
        return (arr.data.device_id, arr.data.mem, arr.data.ptr)

    def _register_new_array(self, arr, dst_actor):
        if self.local_cache:
            self.dist_dict[self._get_array_hash(arr)][dst_actor] = arr
            return arr
        else:
            return arr

    def get_options(self, cluster_entry, cluster_shape):
        node_entry = self.get_cluster_entry(cluster_entry, cluster_shape)
        return {
            "dst_actor": node_entry[0]
        }

    def delete(self, arr):
        if not self.immediate_gc:
            if self.dist_dict is not None:
                del self.dist_dict[self._get_array_hash(arr)]

    def shutdown(self):
        self.dist_dict = None
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()


def get_flatten_id(grid_entry, grid_shape):
    ret = 0
    for i in range(len(grid_entry)):
        dim = 1 if i == len(grid_entry) - 1 else grid_shape[i+1]
        ret = (ret + grid_entry[i]) * dim

    return ret

