from collections import OrderedDict
from math import erf
from numba import jit, njit, int64, float64, deferred_type, optional, types
from numba.experimental import jitclass
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats as stats

from utils import bincount, logtrunc_phi, isin, isin_arr, logsumexp2, logsumexp3


node_type = deferred_type()

spec = OrderedDict()
spec['id'] = int64
spec['parent'] = optional(node_type)
spec['left_child'] = optional(node_type)  # first child
spec['right_child'] = optional(node_type) # last child
spec['sibling'] = optional(node_type)  # next sibling

spec['scope'] = int64[:]
spec['type'] = types.unicode_type
spec['n'] = float64
spec['w'] = optional(float64[:])
spec['logw'] = optional(float64[:])
spec['comparison'] = int64
spec['value'] = float64[:]
spec['mean'] = float64
spec['std'] = float64
spec['a'] = float64
spec['b'] = float64
spec['logcounts'] = optional(float64[:])
spec['p'] = optional(float64[:])
spec['logp'] = optional(float64[:])
spec['upper'] = float64[:]
spec['lower'] = float64[:]


@jitclass(spec)
class Node:
    def __init__(self, parent, scope, type, n):
        self.id = np.random.randint(0, 10000000) # Random identification number
        self.parent = parent
        # initialize parent and left right children as None
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.scope = scope
        self.type = type
        self.n = n
        if parent is not None:
            parent.add_child(self)
        # Sum params
        self.w = None
        self.logw = None
        # Leaf params
        self.value = np.zeros(1, dtype=np.float64)
        self.comparison = -1
        # Gaussian leaf params
        self.mean = 0.
        self.std = 1.
        self.a = -np.Inf
        self.b = np.Inf
        self.p = None
        self.logp = None
        self.logcounts = None
        self.upper = np.ones(len(scope))*(np.Inf)
        self.lower = np.ones(len(scope))*(-np.Inf)

    @property
    def children(self):
        """ A list with all children. """
        children = []
        child = self.left_child
        while child is not None:
            children.append(child)
            child = child.sibling
        return children

    @property
    def nchildren(self):
        return len(self.children)

    def add_sibling(self, sibling):
        self.sibling = sibling

    def add_child(self, child):
        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child
        if self.type == 'S':
            self.reweight()

    def reweight(self):
        children_n = np.array([c.n for c in self.children])
        n = np.sum(children_n)
        if n > 0:
            self.n = n
            self.w = np.divide(children_n.ravel(), self.n)
            self.logw = np.log(self.w.ravel())


node_type.define(Node.class_type.instance_type)


###########################
### INTERFACE FUNCTIONS ###
###########################

@njit
def ProdNode(parent, scope, n):
    return Node(parent, scope, 'P', n)

@njit
def SumNode(parent, scope, n):
    return Node(parent, scope, 'S', n)

@njit
def Leaf(parent, scope, n, value, comparison):
    node = Node(parent, scope, 'L', n)
    fit_indicator(node, value, comparison)
    return node

@njit
def GaussianLeaf(parent, scope, n):
    return Node(parent, scope, 'G', n)

@njit
def MultinomialLeaf(parent, scope, n):
    return Node(parent, scope, 'M', n)

@njit
def UniformLeaf(parent, scope, n, value):
    node = Node(parent, scope, 'U', n)
    node.value = value
    return node


#################################
###    AUXILIARY FUNCTIONS    ###
#################################


def n_nodes(node):
    if node.type in ['L', 'G', 'M']:
        return 1
    if node.type in ['S', 'P']:
        n = 1 + np.sum([n_nodes(c) for c in node.children])
    return n


def delete(node):
    for c in node.children:
        delete(c)
    node.parent = None
    node.left_child = None
    node.right_child = None
    node.sibling = None
    node = None


###########################
###    FIT FUNCTIONS    ###
###########################

@njit
def fit_gaussian(node, data, upper, lower):
    assert node.type == 'G', "Only gaussian leaves fit data."
    node.n = data.shape[0]
    m = np.nanmean(data[:, Int])
    if np.isnan(m):
        node.mean = 0.  # Assuming the data has been standardized
        node.std = np.sqrt(1.)
    else:
        node.mean = m
        if node.n > 1:  # Avoid runtimewarning
            node.std = np.std(data[:, node.scope])
        else:
            node.std = np.sqrt(1.)  # Probably not the best solution here
        node.std = max(np.sqrt(1.), node.std)
        # Compute the tresholds to truncate the Gaussian.
        # The Gaussian has support [a, b]
    node.a = lower
    node.b = upper


@njit
def fit_multinomial(node, data, k):
    assert node.type == 'M', "Node is not a multinomial leaf."
    d = data[~np.isnan(data[:, node.scope].ravel()), :]  # Filter missing
    d = data[:, node.scope].ravel()  # Project to scope
    d = np.asarray(d, np.int64)
    if d.shape[0] > 0:
        counts = bincount(d, k) + 1e-6
        node.logcounts = np.log(counts)
        node.p = counts/(d.shape[0] + k*1e-6)
    else:
        node.p = np.ones(k) * (1/k)
    node.logp = np.log(np.asarray(node.p))


def fit_multinomial_with_counts(node, counts):
    assert node.type == 'M', "Node is not a multinomial leaf."
    node.logcounts = np.log(counts)
    node.p = counts/np.sum(counts)
    node.logp = np.log(node.p)


@njit
def fit_indicator(node, value, comparison):
    node.value = value
    node.comparison = comparison


###########################
### EVALUATE FUNCTIONS  ###
###########################

@njit
def eval_eq(scope, value, evi):
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)-np.Inf
    for i in range(evi.shape[0]):
        if (evi[i, s] == v) or np.isnan(evi[i, s]):
            res[i] = 0
    return res


@njit
def eval_leq(scope, value, evi):
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if evi[i, s] > v:
            res[i] = -np.Inf
    return res


@njit
def eval_g(scope, value, evi):
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if evi[i, s] <= v:
            res[i] = -np.Inf
    return res


@njit
def eval_in(scope, value, evi):
    s, v = scope[0], value[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not isin(evi[i, s], value):
            res[i] = -np.Inf
    return res


@njit
def eval_gaussian(node, evi):
    return logtrunc_phi(evi[:, node.scope].ravel(), node.mean, node.std, node.a, node.b).reshape(-1)


@njit
def eval_m(node, evi):
    s = node.scope[0]
    res = np.zeros(evi.shape[0], dtype=np.float64)
    for i in range(evi.shape[0]):
        if not np.isnan(evi[i, s]):
            res[i] = node.logp[int(evi[i, s])]
    return res


@njit
def compute_batch_size(n_points, n_features):
    maxmem = max(n_points * n_features + (n_points)/10, 10 * 2 ** 17)
    batch_size = (-n_features + np.sqrt(n_features ** 2 + 4 * maxmem)) / 2
    return int(batch_size)


@njit(parallel=True)
def eval_root(node, evi):
    if node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    else:
        return evaluate(node, evi)


@njit(parallel=True)
def eval_root_children(node, evi):
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)
    sizes[:node.nchildren % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])
    logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
    for thread_idx in nb.prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
    return logprs


@njit
def evaluate(node, evi):
    if node.type == 'L':
        if node.comparison == 0:  # IN
            return eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            return eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            return eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            return eval_g(node.scope, node.value, evi)
    elif node.type == 'M':
        return eval_m(node, evi)
    elif node.type == 'G':
        return eval_gaussian(node, evi)
    elif node.type == 'U':
        return np.ones(evi.shape[0]) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        logprs[:, 0] = evaluate(node.children[0], evi)
        nonzero = np.where(logprs[:, 0] != -np.Inf)[0]
        if len(nonzero) > 0:
            for i in nb.prange(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, i] = evaluate(node.children[i], evi[nonzero, :])
        return np.sum(logprs, axis=1)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
        for i in nb.prange(node.nchildren):
            logprs[:, i] = evaluate(node.children[i], evi) + node.logw[i]
        res = logsumexp2(logprs, axis=1)
        return res
    return np.zeros(evi.shape[0])


@njit(parallel=True)
def eval_root_class(node, evi, class_var, n_classes, naive):
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, node.nchildren // n_threads, dtype=np.int32)
    sizes[:node.nchildren % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int32)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])
    logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
    for thread_idx in nb.prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            if naive:
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive)  # no weights here
            else:
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
    return logprs


@njit
def evaluate_class(node, evi, class_var, n_classes, naive):
    if node.type == 'L':
        res = np.zeros((evi.shape[0], 1))
        if node.comparison == 0:  # IN
            res[:, 0] = eval_in(node.scope, node.value.astype(np.int64), evi)
        elif node.comparison == 1:  # EQ
            res[:, 0] = eval_eq(node.scope, node.value.astype(np.float64), evi)
        elif node.comparison == 3:  # LEQ
            res[:, 0] = eval_leq(node.scope, node.value, evi)
        elif node.comparison == 4:  # G
            res[:, 0] = eval_g(node.scope, node.value, evi)
        return res
    elif node.type == 'M':
        if isin(class_var, node.scope):
            if naive:
                return np.zeros((evi.shape[0], n_classes)) + node.logcounts
            else:
                return np.zeros((evi.shape[0], n_classes)) + node.logp
        if naive:
            return np.zeros((evi.shape[0], 1))
        res = np.zeros((evi.shape[0], 1))
        res[:, 0] = eval_m(node, evi)
        return res
    elif node.type == 'G':
        res = np.zeros((evi.shape[0], 1))
        if naive:
            return res
        res[:, 0] = eval_gaussian(node, evi)
        return res
    elif node.type == 'U':
        if naive:
            return np.zeros((evi.shape[0], 1))
        return np.ones((evi.shape[0], 1)) * node.value
    elif node.type == 'P':
        logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
        logprs[:, :, 0] = evaluate_class(node.children[0], evi, class_var, n_classes, naive)
        nonzero = ~np.isinf(logprs[:, 0, 0])
        if np.sum(nonzero) > 0:
            for i in range(1, node.nchildren):
                # Only evaluate nonzero examples to save computation
                logprs[nonzero, :, i] = evaluate_class(node.children[i], evi[nonzero, :], class_var, n_classes, naive)
        return np.sum(logprs, axis=2)
    elif node.type == 'S':
        logprs = np.zeros((evi.shape[0], n_classes, node.nchildren), dtype=np.float64)
        if naive:
            for i in range(node.nchildren):
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive)  # no weights here
        else:
            for i in range(node.nchildren):
                logprs[:, :, i] = evaluate_class(node.children[i], evi, class_var, n_classes, naive) + node.logw[i]
        return logsumexp3(logprs, axis=2)
    return np.zeros((evi.shape[0], 1))


def evaluate_bottom_up(spn, node, evi):
    order = spn.order
    logprobs = np.zeros((evi.shape[0], len(order))) -1
    for node in reversed(order[node.id:]):
        if node.type == 'P':
            logprobs[:, node.id] = 0
            for child in node.children:
                logprobs[:, node.id] = logprobs[:, node.id] + logprobs[:, child.id]
        elif node.type == 'S':
            logprs = np.zeros((evi.shape[0], node.nchildren), dtype=np.float64)
            for i, child in enumerate(node.children):
                logprs[:, i] = logprobs[:, child.id] + node.logw[i]
            logprobs[:, node.id] = logsumexp(logprs, axis=1)
        elif node.type == 'M':
            logprobs[:, node.id] = eval_m(node, evi)
        elif node.type == 'U':
            logprobs[:, node.id] = node.value
        elif node.type == 'L':
            if node.comparison == Comparison.IN:
                logprobs[:, node.id] = eval_in(node.scope[0], node.value, evi)
            elif node.comparison == Comparison.EQ:
                logprobs[:, node.id] = eval_eq(node.scope[0], node.value[0], evi)
            elif node.comparison == Comparison.LEQ:
                logprobs[:, node.id] = eval_leq(node.scope[0], node.value, evi)
            elif node.comparison == Comparison.G:
                logprobs[:, node.id] = eval_g(node.scope[0], node.value, evi)
    return logprobs[:, node.id]
