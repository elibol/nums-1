import time
from typing import List, Union

import numpy as np
import ray

from nums.core import settings
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.application_manager import instance


def init_rpcs(app: ArrayApplication):
    app.system.register("fused_objective", fused_objective_rpc, {})
    app.system.register("fused_grad_diff", fused_grad_diff, {})
    app.system.register("fused_bt_test", fused_bt_test, {})


def fused_objective_rpc(eta, y):
    mu = 1.0 / (1.0 + np.exp(-eta))
    return - np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu + 1e-14))


def fused_objective(app: ArrayApplication, X, y, theta):
    eta: BlockArray = X @ theta
    res: BlockArray = BlockArray(eta.grid.copy(), app.system)
    for grid_entry in eta.grid.get_entry_iterator():
        res.blocks[grid_entry].oid = app.system.call("fused_objective",
                                                     eta.blocks[grid_entry].oid,
                                                     y.blocks[grid_entry].oid,
                                                     syskwargs={
                                                         "grid_entry": grid_entry,
                                                         "grid_shape": eta.grid.grid_shape
                                                     })
    return app.sum(res)


def fused_grad_diff(eta, y):
    mu = 1.0 / (1.0 + np.exp(-eta))
    return mu - y


def fused_grad(app, X, y, theta, eta=None):
    if eta is None:
        eta = X @ theta
    diff: BlockArray = BlockArray(eta.grid.copy(), app.system)
    for grid_entry in eta.grid.get_entry_iterator():
        diff.blocks[grid_entry].oid = app.system.call("fused_grad_diff",
                                                      eta.blocks[grid_entry].oid,
                                                      y.blocks[grid_entry].oid,
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": eta.grid.grid_shape
                                                      })
    return X.T @ diff


def forward(app, X, theta, eta=None):
    if eta is None:
        eta = X @ theta
    mu = app.one / (app.one + app.exp(-eta))
    return mu


def fused_bt_test(f_next, f_val, c, alpha, grad_p_inner):
    return np.isnan(f_next) or f_next > f_val + c * alpha * grad_p_inner


# Not sure this can be improved any further...
def bt_linesearch(app,
                  X, y, theta,
                  grad, p,
                  rho=1.e-1, init_alpha=1.0, c=1e-4, min_alpha=1e-10):

    def f(theta_prime):
        return fused_objective(app, X, y, theta_prime)

    alpha = init_alpha
    f_val = f(theta)
    f_next = f(theta + alpha * p)
    remote_c = app.system.put(c)
    while True:
        grad_p_inner = grad.T @ p

        # Original imp.
        # if not (app.isnan(f_next) or f_next > f_val + c * alpha * grad_p_inner):
        #     break

        # Remote imp.
        # TODO: Get this working if we're slow.
        # remote_alpha = app.system.put(alpha)
        # if app.system.get(app.system.call("fused_bt_test",
        #                                   f_next.flattened_oids()[0],
        #                                   f_val.flattened_oids()[0],
        #                                   remote_c,
        #                                   remote_alpha,
        #                                   grad_p_inner.flattened_oids()[0],
        #                                   syskwargs={
        #                                       "grid_entry": tuple([0]*len(grad_p_inner.grid.grid_shape)),
        #                                       "grid_shape": grad_p_inner.grid.grid_shape
        #                                   })):
        #     break

        # Local imp.
        f_val_local = f_val.get()
        f_next_local = f_next.get()
        grad_p_inner_local = grad_p_inner.get()
        if not (np.isnan(f_next_local) or
                f_next_local > f_val_local + c * alpha * grad_p_inner_local):
            break
        alpha *= rho
        if alpha < min_alpha:
            return min_alpha
        # print("btls step alpha=%s" % alpha)
        f_next = f(theta + alpha * p)
    return alpha


class LBFGSMemory(object):

    def __init__(self, k, s, y):
        self.k = k
        self.s = s
        self.y = y
        ys_inner = s.T @ y
        self.rho = 1.0 / ys_inner
        self.gamma = ys_inner / y.T @ y


@ray.remote(num_cpus=1)
class LBFGSMemoryActor(object):
    # Hessian etc. are not that big, so ok to use actor.

    def __init__(self, m, dtype):
        self.identity = None
        self.dtype = dtype
        self.memory: Union[List[LBFGSMemory], List[None]] = [None]*m
        self.k = 0
        self.m = m
        self.theta = None
        self.g = None
        self.p = None
        self.H = None

    def init(self, *data, theta_size):
        self.theta = np.concatenate(data[:theta_size])
        self.g = np.concatenate(data[theta_size:])
        self.identity = np.eye(self.g.size, dtype=self.dtype)

    def reset(self):
        self.memory = [None]*self.m

    def get_H(self):
        if self.k == 0:
            return self.identity
        else:
            mem: LBFGSMemory = self.memory[-1]
            assert mem.k == self.k-1
            return mem.gamma * self.identity

    def _get_p(self, H, g):
        q = g.copy()
        forward_vars = []
        for i in range(-1, -self.m-1, -1):
            mem_i: LBFGSMemory = self.memory[i]
            if mem_i is None:
                break
            alpha = mem_i.rho * mem_i.s.T @ q
            q -= alpha * mem_i.y
            forward_vars.insert(0, (alpha, mem_i))
        r = H @ q
        for alpha, mem_i in forward_vars:
            beta = mem_i.rho * mem_i.y.T @ r
            r += mem_i.s * (alpha - beta)
        return r

    def fetch_H(self):
        return 0 if self.memory[-1] is None else self.memory[-1].gamma

    def update_p(self):
        self.H = self.get_H()
        self.p = - self._get_p(self.H, self.g)
        return self.p

    def update_mem(self, *data, theta_size):
        next_theta = np.concatenate(data[:theta_size])
        next_g = np.concatenate(data[theta_size:])
        theta_diff = next_theta - self.theta
        grad_diff = next_g - self.g
        mem: LBFGSMemory = LBFGSMemory(k=self.k, s=theta_diff, y=grad_diff)
        self.memory.append(mem)
        self.memory.pop(0)
        self.theta = next_theta
        self.g = next_g
        self.k += 1

    def fetch_g(self):
        return self.g

    def converged(self, g):
        return self.app.sqrt(g.T @ g) < self.thresh


class LBFGS(object):

    def __init__(self, app: ArrayApplication,
                 m=3, max_iter=100,  thresh=1e-5, dtype=np.float32):
        self.app: ArrayApplication = app
        self.m = m
        self.k = 0
        self.max_iter = max_iter
        self.thresh = thresh
        self.dtype = dtype
        # Place this on node 0.
        options = self.app.system.get_options(tuple([0]*len(settings.cluster_shape)),
                                              settings.cluster_shape)
        self.lbfgs_mem: LBFGSMemoryActor = LBFGSMemoryActor.options(options).remote(m, dtype)

    def execute(self, X, y, theta):

        if self.k != 0:
            raise Exception("Unexpected state.")

        g: BlockArray = fused_grad(self.app, X, y, theta)

        data = theta.flattened_oids()
        theta_size = len(data)
        data += g.flattened_oids()
        self.lbfgs_mem.init.remote(*data, theta_size=theta_size)

        next_g = None
        next_theta = None
        while self.k < self.max_iter:

            p = BlockArray.from_oid(self.lbfgs_mem.update_p.remote(),
                                    shape=g.shape,
                                    dtype=self.dtype,
                                    system=self.app.system).reshape(block_shape=g.block_shape)

            init_alpha = min(1.0, 10**(self.k-self.max_iter/2))
            alpha = bt_linesearch(self.app, X, y,
                                  theta, g, p,
                                  rho=1e-2,
                                  init_alpha=init_alpha,
                                  c=1e-4,
                                  min_alpha=1e-30)
            print(alpha)
            next_theta = theta + alpha * p
            if self.k + 1 >= self.max_iter:
                # Terminate immediately if this is the last iteration.
                theta = next_theta
                break
            next_g = fused_grad(self.app, X, y, next_theta)

            data = next_theta.flattened_oids()
            data += next_g.flattened_oids()
            self.lbfgs_mem.update_mem.remote(*data, theta_size=theta_size)

            theta = next_theta
            g = next_g
            self.k += 1

        # Reset vars.
        self.k = 0
        self.lbfgs_mem.reset.remote()
        return theta


def logistic(app, X, y, max_iter, m):
    Xc = app.concatenate([X, app.ones(shape=(X.shape[0], 1),
                                      block_shape=(X.block_shape[0], 1),
                                      dtype=X.dtype)],
                         axis=1,
                         axis_block_size=X.block_shape[1])
    theta = app.zeros((Xc.shape[1],), (Xc.block_shape[1],), dtype=Xc.dtype)
    lbfgs_optimizer = LBFGS(app, m=m, max_iter=max_iter, dtype=Xc.dtype)
    theta = lbfgs_optimizer.execute(Xc, y, theta)
    return forward(app, Xc, theta)


def sample_set(app: ArrayApplication):
    shape = (500, 10)
    block_shape = (100, 10)
    rs = app.random_state(1337)
    X1 = rs.normal(loc=5.0, shape=shape, block_shape=block_shape)
    y1 = app.zeros(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X2 = rs.normal(loc=10.0, shape=shape, block_shape=block_shape)
    y2 = app.ones(shape=(shape[0],), block_shape=(block_shape[0],), dtype=int)
    X = app.concatenate([X1, X2], axis=0)
    y = app.concatenate([y1, y2], axis=0)
    return X, y


def load_set(app: ArrayApplication, read_func, dataset):
    X = read_func("%s_X" % dataset)
    y = read_func("%s_y" % dataset)
    return X, y


def execute(dataset, cluster_shape, address, use_s3):

    settings.cluster_shape = cluster_shape
    ray.init(address=address)
    app: ArrayApplication = instance()
    init_rpcs(app)
    time.sleep(1.0)

    start_time = time.time()
    read_func = app.read_s3 if use_s3 else app.read_fs
    # X, y = load_set(app, read_func, dataset)
    X, y = sample_set(app)
    y_pred_proba = logistic(app, X, y, max_iter=10, m=3)
    print("scheduling submitted.")
    y_pred = (y_pred_proba > 0.5).astype(np.float32)
    print("prediction submitted.")
    error = (app.sum(app.abs(y - y_pred)) / X.shape[0]).astype(np.float32).get()
    total_time = time.time() - start_time
    print("opt", "lbfgs")
    print("total time", total_time)
    print("error (1-accuracy)", error)
    # print("norm", model.grad_norm_sq(X, y).get())
    # print("objective", model.objective(X, y).get())
    return total_time, float(error)


if __name__ == "__main__":
    execute(dataset=None, cluster_shape=(1, 1),
            address=None, use_s3=False)
