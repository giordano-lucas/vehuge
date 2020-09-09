from collections import OrderedDict
from functools import reduce
from math import floor
import numba as nb
from numba import jit, njit, int16, int32, float32, int64, float64, optional, prange, deferred_type, types, boolean
from numba.typed import List
from numba.experimental import jitclass
import numba as nb
import numpy as np
import operator
import random
from scipy import stats
import time
from tqdm import tqdm

from learning import LearnSPN, fit
from nb_nodes import SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf, UniformLeaf, fit_multinomial, fit_multinomial_with_counts, fit_gaussian
from spn import SPN
from utils import bincount, learncats, purity, isin_nb, resample_strat
from split import find_best_split, Split


@njit(parallel=True)
def evaluate(node, X, n_classes=2):
    n_samples = X.shape[0]
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    res = np.empty((X.shape[0], n_classes), dtype=np.float64)

    if n_samples < n_threads:
        for i in range(n_samples):
            res[i, :] = evaluate_instance(node, X[i, :])
        return res

    sizes = np.full(n_threads, n_samples // n_threads, dtype=np.int64)
    sizes[:n_samples % n_threads] += 1
    offset_in_buffers = np.zeros(n_threads, dtype=np.int64)
    offset_in_buffers[1:] = np.cumsum(sizes[:-1])

    for thread_idx in prange(n_threads):
        start = offset_in_buffers[thread_idx]
        stop = start + sizes[thread_idx]
        for i in range(start, stop):
            res[i, :] = evaluate_instance(node, X[i, :])
    return res

@njit
def evaluate_instance(node, X):
    s = node.split
    if s is None:
        # Leaf Node
        return node.counts
    if not np.isnan(X[s.var]):
        # Observed split variable
        if s.type == 'num':
            go_left = X[s.var] <= s.threshold[0]
        else:
            go_left = np.any(isin_nb(np.array(X[s.var]), s.threshold))
    else:
        # Missing split variable; look for surrogate split
        var = -1
        for i in range(len(s.surr_var)):
            if not np.isnan(X[s.surr_var[i]]):
                var = s.surr_var[i]
                thr = s.surr_thr[i]
                left = s.surr_go_left[i]
                break
        if var == -1:
            # If all missing, take a side at random
            go_left = s.surr_blind
        elif X[var] <= thr:
            go_left = left
        else:
            go_left = ~left
    if go_left:
        return evaluate_instance(node.left_child, X)
    else:
        return evaluate_instance(node.right_child, X)

@njit
def build_tree(tree, parent, counts, ordered_ids):
    root = TreeNode(0, counts, parent, ordered_ids, False)
    queue = List()
    queue.append(root)
    n_nodes = 1
    while len(queue) > 0:
        node = queue.pop(0)
        split = find_best_split(node, tree)
        if split is not None:
            node.split = split
            left_child = TreeNode(n_nodes, split.left_counts, node, split.left_ids, False)
            node.left_child = left_child
            queue.append(left_child)
            n_nodes += 1
            right_child = TreeNode(n_nodes, split.right_counts, node, split.right_ids, False)
            node.right_child = right_child
            queue.append(right_child)
            n_nodes += 1
        else:
            node.isleaf = True
        tree.depth = max(tree.depth, node.depth)
    return root, n_nodes


@njit(parallel=True)
def build_forest(X, y, n_estimators, bootstrap, ncat, imp_measure, min_samples_split, min_samples_leaf, max_features, max_depth, surrogate):
    n_samples = X.shape[0]
    n_classes = np.max(y)+1  # We assume ordinals from 0, 1, 2, ..., max(y)
    n_threads = nb.config.NUMBA_DEFAULT_NUM_THREADS
    estimators = [Tree(ncat, imp_measure, min_samples_split, min_samples_leaf, max_features, max_depth, surrogate) for i in range(n_estimators)]

    if n_estimators < n_threads:
        for i in prange(n_estimators):
            Xtree_, ytree_ = resample_strat(X, y, n_classes)
            estimators[i].fit(Xtree_, ytree_)
    else:
        sizes = np.full(n_threads, n_estimators // n_threads, dtype=np.int64)
        sizes[:n_estimators % n_threads] += 1
        offset_in_buffers = np.zeros(n_threads, dtype=np.int64)
        offset_in_buffers[1:] = np.cumsum(sizes[:-1])

        for thread_idx in prange(n_threads):
            start = offset_in_buffers[thread_idx]
            stop = start + sizes[thread_idx]
            for i in range(start, stop):
                Xtree_, ytree_ = resample_strat(X, y, n_classes)
                estimators[i].fit(Xtree_, ytree_)
    return estimators


@njit
def add_split(tree_node, spn_node, ncat, root=False):

    split_col = tree_node.split.var
    split_value = tree_node.split.threshold
    n_points_left = len(tree_node.split.left_ids)
    n_points_right = len(tree_node.split.right_ids)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    scope = np.arange(len(ncat), dtype=np.int64)
    if root:
        sumnode = spn_node
    else:
        sumnode = SumNode(scope=scope, parent=spn_node, n=n_points_left+n_points_right+lp)

    upper1 = spn_node.upper.copy()
    lower1 = spn_node.lower.copy()
    upper2 = spn_node.upper.copy()
    lower2 = spn_node.lower.copy()

    if ncat[split_col] > 1:
        cat = np.arange(ncat[split_col], dtype=np.float64)
        mask = ~isin_nb(cat, split_value)
        out_split_value = cat[mask]

        upper1[split_col] = len(split_value)  # upper: number of variables in
        lower1[split_col] = len(out_split_value)  # lower: number of variables out
        p1 = ProdNode(scope=scope, parent=sumnode, n=n_points_left+lp)
        p1.upper, p1.lower = upper1, lower1
        ind1 = Leaf(scope=np.array([split_col]), parent=p1, n=n_points_left+lp, value=split_value, comparison=0)  # Comparison IN

        upper2[split_col] = len(out_split_value)  # upper: number of variables in
        lower2[split_col] = len(split_value)  # lower: number of variables out
        p2 = ProdNode(scope=scope, parent=sumnode, n=n_points_right+lp)
        p2.upper, p2.lower = upper2, lower2
        ind2 = Leaf(scope=np.array([split_col]), parent=p2, n=n_points_right+lp, value=out_split_value, comparison=0)  # Comparison IN
    else:
        upper1[split_col] = min(split_value[0], upper1[split_col])
        p1 = ProdNode(scope=scope, parent=sumnode, n=n_points_left+lp)
        p1.upper, p1.lower = upper1, lower1
        ind1 = Leaf(scope=np.array([split_col]), parent=p1, n=n_points_left+lp, value=split_value, comparison=3)  # Comparison <=

        lower2[split_col] = max(split_value[0], lower2[split_col])
        p2 = ProdNode(scope=scope, parent=sumnode, n=n_points_right+lp)
        p2.upper, p2.lower = upper2, lower2
        ind2 = Leaf(scope=np.array([split_col]), parent=p2, n=n_points_right+lp, value=split_value, comparison=4)  # Comparison >

    return p1, p2


def add_dist(tree_node, spn_node, data, ncat, learnspn, uniformleaves, uni_everywhere):
    counts = tree_node.counts
    n_points = len(tree_node.idx)
    data_leaf = data[tree_node.idx, :]
    scope = np.arange(data.shape[1], dtype=np.int64)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    upper, lower = spn_node.upper, spn_node.lower

    if (learnspn) and (n_points >= 30):
        # The class variable is modeled independently as multinomial data
        learner = LearnSPN(ncat=ncat, classcol=None)
        fit(learner, data_leaf, last_node=spn_node)
    elif (uniformleaves in ['const', 'per_leaf']) and ((n_points < 10) or uni_everywhere):
        classleaf = MultinomialLeaf(scope=np.array([scope[-1]]), parent=spn_node, n=n_points)
        fit_multinomial(classleaf, data_leaf, int(ncat[data.shape[1]-1]))
        # Uniform distribution defined on [-3, 3]
        # Assuming mean 0 and std 1, this should cover most of the data
        if uniformleaves == 'per_leaf':
            vol = np.array([min(upper[i] - lower[i], 6) if ncat[i]==1 else upper[i] for i in range(data.shape[1]-1)])
        else:
            vol = np.array([6 if nc==1 else nc for nc in ncat])
        eps = np.log([reduce(operator.mul, 1/vol)]).astype(float)
        uniformleaf = UniformLeaf(scope=scope[:-1], parent=spn_node, n=data_leaf.shape[0]+lp, value=eps)
        return None
    else:
        for var in scope:
            if ncat[var] > 1:
                if var == len(ncat)-1:
                    leaf = MultinomialLeaf(scope=np.array([var]), parent=spn_node, n=n_points+lp)
                    fit_multinomial(leaf, data_leaf, int(ncat[var]))
                else:
                    leaf = MultinomialLeaf(scope=np.array([var]), parent=spn_node, n=n_points+lp)
                    fit_multinomial(leaf, data_leaf, int(ncat[var]))
            else:
                leaf = GaussianLeaf(scope=np.array([var]), parent=spn_node, n=n_points+lp)
                fit_gaussian(leaf, data_leaf, upper[var], lower[var])
        return None


def tree2spn(tree, learnspn=False, uniformleaves=None, uni_everywhere=False):
    scope = np.array([i for i in range(tree.X.shape[1]+1)]).astype(int)
    data = np.concatenate([tree.X,
                           np.expand_dims(tree.y, axis=1)], axis=1)
    ncat = np.array(tree.ncat)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    upper = ncat.copy().astype(np.float64)
    upper[upper == 1] = np.Inf
    lower = ncat.copy().astype(np.float64)
    lower[ncat == 1] = -np.Inf
    lower = np.ones(data.shape[1])*(-np.Inf)
    classcol = len(ncat)-1

    spn = SPN()
    spn_node = SumNode(scope=scope, parent=None, n=data.shape[0]+lp)
    spn.root = spn_node
    spn.ncat = ncat

    tree_queue = [tree.root]
    spn_queue = [spn.root]

    root = True
    while len(tree_queue) > 0:
        tree_node = tree_queue.pop(0)
        spn_node = spn_queue.pop(0)

        if tree_node.isleaf:
            add_dist(tree_node, spn_node, data, ncat, learnspn, uniformleaves, uni_everywhere)
        else:
            p_left, p_right = add_split(tree_node, spn_node, ncat, root)
            tree_queue.extend([tree_node.left_child, tree_node.right_child])
            spn_queue.extend([p_left, p_right])
        root = False

    return spn


def delete(node):
    if node.left_child is not None:
        delete(node.left_child)
        node.left_child = None
    if node.right_child is not None:
        delete(node.right_child)
        node.right_child = None
    node.parent = None
    node = None


def delete_tree(tree):
    delete(tree.root)


node_type = deferred_type()

@jitclass([
    ('id', int64),
    ('counts', int64[:]),
    ('idx', int64[:]),
    ('split', optional(Split.class_type.instance_type)),
    ('parent', optional(node_type)),
    ('left_child', optional(node_type)),
    ('right_child', optional(node_type)),
    ('isleaf', optional(nb.boolean)),
    ('depth', int16),
])
class TreeNode:
    def __init__(self, id, counts, parent, idx, isleaf):
        self.id = id
        self.counts = counts
        self.parent = parent
        self.idx = idx
        self.isleaf = isleaf
        self.split = None
        self.left_child = None
        self.right_child = None
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

node_type.define(TreeNode.class_type.instance_type)


@jitclass([
    ('X', optional(float64[:,:])),
    ('y', optional(int64[:])),
    ('ncat', optional(int64[:])),
    ('scope', optional(int64[:])),
    ('imp_measure', types.string),
    ('min_samples_leaf', int64),
    ('min_samples_split', int64),
    ('n_classes', int64),
    ('max_features', optional(int64)),
    ('n_nodes', int64),
    ('root', TreeNode.class_type.instance_type),
    ('depth', int16),
    ('max_depth', int64),
    ('surrogate', boolean),
])
class Tree:
    def __init__(self, ncat=None, imp_measure='gini', min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=1e6, surrogate=False):
        self.X = None
        self.y = None
        self.ncat = ncat
        self.scope = None
        self.imp_measure = imp_measure
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_nodes = 0
        self.n_classes = 0
        self.depth = 0
        self.max_depth = max_depth
        self.root = TreeNode(0, np.empty(0, dtype=np.int64), None, np.empty(0, dtype=np.int64), False)
        self.surrogate = surrogate

    def fit(self, X, y):
        """
            Parameters
            ----------
            X, y: numpy array
                The data divided into independent (X) and independent (y) variables.
        """
        self.X = X
        self.y = y
        self.n_classes = np.max(y)+1
        self.n_nodes = 0
        if self.max_features is None:
            self.max_features = X.shape[1]

        counts = bincount(y, self.ncat[-1])
        ordered_ids = np.arange(X.shape[0], dtype=np.int64)
        self.root, self.n_nodes = build_tree(self, None, counts, ordered_ids)

    def get_node(self, id):
        """ Fetchs node by its id. """
        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.split is not None:
                queue.extend([node.left_child, node.right_child])
        print("Node %d is not part of the network.", id)

    def predict(self, X):
        root = self.root
        counts = evaluate(root, X, self.n_classes)
        res = np.empty(counts.shape[0], dtype=np.float64)
        for i in range(counts.shape[0]):
            res[i] = np.argmax(counts[i, :])
        return res

    def predict_proba(self, X):
        counts = evaluate(self.root, X, self.n_classes)
        res = np.empty(counts.shape)
        for i in range(counts.shape[0]):
            res[i, :] = (counts[i, :])/np.sum(counts[i, :])
        return res

    def get_node_of_type(self, type):
        """ Fetchs all nodes of a given type. """
        queue = [self.root]
        res = []
        while queue != []:
            node = queue.pop(0)
            if node.isleaf:
                if type == 'L':
                    res.append(node)
            else:
                queue.extend([node.left_child, node.right_child])
                if type == 'D':
                    res.append(node)
        return res


class RandomForest:
    def __init__(self, n_estimators=100, imp_measure='gini', min_samples_split=2,
                 min_samples_leaf=1, max_features=None, bootstrap=True,
                 ncat=None, max_depth=1e6, surrogate=False):

        self.n_estimators = n_estimators
        self.imp_measure = imp_measure
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.ncat = ncat.astype(np.int64)
        self.max_features = max_features
        self.max_depth = max_depth
        self.surrogate = surrogate

    def fit(self, X, y):
        y = y.astype(np.int64)
        n = X.shape[0]
        if self.max_features is None:
            self.max_features = min(floor(X.shape[1]/3), X.shape[1])
        self.scope = np.array([i for i in range(X.shape[1]+1)])
        self.estimators = build_forest(X, y, self.n_estimators, self.bootstrap,
                                       self.ncat, self.imp_measure,
                                       self.min_samples_split,
                                       self.min_samples_leaf,
                                       self.max_features, self.max_depth,
                                       self.surrogate)

    def tospn(self, X_train=None, y_train=None, learnspn=False, uniformleaves=None, uni_everywhere=False):
        """ Returns an SPN Model. """
        spn = SPN()
        spn.root = SumNode(scope=self.scope, parent=None, n=1)
        if (X_train is not None) and (y_train is not None):
            print("Fitting SPN with all data.")
            for estimator in self.estimators:
                tree_spn = tree2spn(estimator, learnspn=learnspn, uniformleaves=uniformleaves, uni_everywhere=uni_everywhere)
                spn.root.add_child(tree_spn.root)
        else:
            print("Fitting SPN with bootstrap data only.")
            for estimator in self.estimators:
                tree_spn = tree2spn(estimator, learnspn=learnspn, uniformleaves=uniformleaves, uni_everywhere=uni_everywhere)
                spn.root.add_child(tree_spn.root)
        spn.ncat = tree_spn.ncat
        return spn

    def predict(self, X, vote=True):
        if vote:
            votes = np.zeros(shape=(X.shape[0], self.n_estimators))
            for i, estimator in enumerate(self.estimators):
                votes[:, i] = estimator.predict(X)
            return stats.mode(votes, axis=1)[0].reshape(-1)
        else:
            probas = np.zeros(shape=(X.shape[0], self.n_estimators, self.ncat[-1]))
            for i, estimator in enumerate(self.estimators):
                probas[:, i] =  estimator.predict_proba(X)
            return np.mean(probas, axis=1)

    def delete(self):
        for est in self.estimators:
            delete_tree(est)
        self.estimators = None
