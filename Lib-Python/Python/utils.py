from collections import OrderedDict
from math import erf, floor
from munkres import Munkres
import numba as nb
from numba import jit, njit, prange, jitclass
import numpy as np
import random
import scipy.stats as stats
from statsutils import chi_test, kruskal, kendalltau

def str2bool(v):
    """ Converts a string to a boolean value. """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@njit
def resample_strat(x, y, n_classes):
    idx = np.arange(x.shape[0], dtype=np.int64)
    counts = bincount(y, n_classes)
    selected_idx = np.empty(0, dtype=np.int64)
    for i in range(n_classes):
        s = np.random.choice(idx[y==i], counts[i], replace=True)
        selected_idx = np.concatenate((selected_idx, s))
    return x[selected_idx, :], y[selected_idx]


@njit
def in1d_vec_nb(matrix, index_to_remove):
  #matrix and index_to_remove have to be numpy arrays
  #if index_to_remove is a list with different dtypes this
  #function will fail

    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    index_to_remove_set = set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i] = True
        else:
            out[i] = False
    return out


@njit
def in1d_scal_nb(matrix, index_to_remove):
    # matrix and index_to_remove have to be numpy arrays
    # if index_to_remove is a list with different dtypes this
    # function will fail
    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if (matrix[i] == index_to_remove):
            out[i] = True
        else:
            out[i] = False
        return out


@njit
def isin_nb(matrix_in, index_to_remove):
    # both matrix_in and index_to_remove have to be a np.ndarray
    # even if index_to_remove is actually a single number
    shape = matrix_in.shape
    if index_to_remove.shape == ():
        res = in1d_scal_nb(np.ravel(matrix_in), index_to_remove.take(0))
    else:
        res = in1d_vec_nb(np.ravel(matrix_in), index_to_remove)

    return res.reshape(shape)


@njit(fastmath=True)
def bincount(data, n):
    counts = np.zeros(n, dtype=np.int64)
    for j in range(n):
        counts[j] = np.sum(data==j)
    return counts


@njit
def Phi(x, mean, std):
    """ Cumulative distribution for normal distribution defined by mean and std. """
    return .5*(1.0 + erf((x-mean) / (std*np.sqrt(2.0))))


@njit
def phi(x, mean, std):
    """ Cumulative distribution for normal distribution defined by mean and std. """
    denom = np.sqrt(2*np.pi)*std
    num = np.exp(-.5*((x - mean) / std)**2)
    return num/denom


@njit
def logtrunc_phi(x, loc, scale, a, b):
    res = np.ones(x.shape[0], dtype=np.float64)
    denom = (Phi(b, loc, scale) - Phi(a, loc, scale))
    for i in range(x.shape[0]):
        if (x[i] < a) or (x[i] > b):
            res[i] = -np.Inf
        elif not np.isnan(x[i]):
            res[i] = np.log(phi(x[i], loc, scale)/denom)
    return res


@njit(fastmath=True)
def isin(a, b):
    """ Returns True if a in b. """
    for bi in b:
        if (bi == a):
            return True
    return False


@njit(fastmath=True)
def isin_arr(arr, b):
    res = np.empty(arr.shape[0], dtype=nb.boolean)
    for i in nb.prange(arr.shape[0]):
        res[i] = isin(arr[i], b)
    return res


@njit
def lse(a):
    result = 0.0
    largest_in_a= a[0]
    for i in range(a.shape[0]):
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    if largest_in_a == -np.inf:
        return a[0]
    for i in range(a.shape[0]):
        result += np.exp(a[i] - largest_in_a)
    return np.log(result) + largest_in_a


@njit
def logsumexp2(a, axis):
    assert a.ndim == 2, 'Wrong logsumexp method.'
    if axis == 0:
        res = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            res[i] = lse(a[:, i].ravel())
    elif axis == 1:
        res = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            res[i] = lse(a[i, :].ravel())
    return res


@njit
def logsumexp3(a, axis):
    assert a.ndim == 3, 'Wrong logsumexp method.'
    if axis == 0:
        res = np.zeros((a.shape[1], a.shape[2]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                res[i, j] = lse(a[:, i, j].ravel())
    elif axis == 1:
        res = np.zeros((a.shape[0], a.shape[2]))
        for i in range(a.shape[0]):
            for j in range(a.shape[2]):
                res[i, j] = lse(a[i, :, j].ravel())
    elif axis == 2:
        res = np.zeros((a.shape[0], a.shape[1]))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                res[i, j] = lse(a[i, j, :].ravel())
    return res


@njit
def hstack(l):
    n_arrays = len(l)  # number of arrays to stack
    m = l[0].shape[0]
    n = l[0].shape[1]
    shape = (m, n*n_arrays)  # shape of desired array
    res = np.zeros(shape)
    for i in range(n_arrays):
        assert l[i].shape == (m, n), "Shape mismatch among arrays."
        for j in range(n):
            res[:, n*i+j] = l[i][:, j]
    return res


def flat(list_of_tuples):
    return set(p for pair in list_of_tuples for p in pair)


def pairwise(iterable):
    """ Produce pairs of elements in an iterable. """
    for i1, l1 in enumerate(iterable[:-1]):
        for i2, l2 in enumerate(iterable[i1+1:]):
            yield l1, l2


# def chi_test(var1, var2):
#     """ Computes a chi-squared test for two variables. """
#     obs = ~np.isnan(var1*var2)
#     result = np.array([[sum((var1 == cat1) & (var2 == cat2))
#                for cat2 in np.unique(var2[obs])]
#                for cat1 in np.unique(var1[obs])])
#     _, v, _, _ = stats.chi2_contingency(result)
#     return v


def learncats(data, classcol=None, continuous_ids=[]):
    """
        Learns the number of categories in each variable and standardizes the data.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        classcol: int
            The column index of the class variables (if any).
        continuous_ids: list of ints
            List containing the indices of known continuous variables. Useful for
            discrete data like age, which is better modeled as continuous.

        Returns
        -------
        ncat: numpy m
            The number of categories of each variable. One if the variable is
            continuous.
    """
    data = data.copy()
    ncat = np.ones(data.shape[1])
    if not classcol:
        classcol = data.shape[1]-1
    for i in range(data.shape[1]):
        if i != classcol and (i in continuous_ids or is_continuous(data[:, i])):
            continue
        else:
            data[:, i] = data[:, i].astype(int)
            ncat[i] = max(data[:, i]) + 1
    return ncat


def get_stats(data, ncat=None):
    """
        Compute univariate statistics for continuous variables.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.

        Returns
        -------
        data: numpy n x m
            The normalized data.
        maxv, minv: numpy m
            The maximum and minimum values of each variable. One and zero, resp.
            if the variable is categorical.
        mean, std: numpy m
            The mean and standard deviation of the variable. Zero and one, resp.
            if the variable is categorical.

    """
    data = data.copy()
    maxv = np.ones(data.shape[1])
    minv = np.zeros(data.shape[1])
    mean = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    if ncat is not None:
        for i in range(data.shape[1]):
            if ncat[i] == 1:
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    else:
        for i in range(data.shape[1]):
            if is_continuous(data[:, i]):
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    return data, maxv, minv, mean, std


def normalize_data(data, maxv, minv):
    """
        Normalizes the data given the maximum and minimum values of each variable.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        maxv, minv: numpy m
            The maximum and minimum values of each variable. One and zero, resp.
            if the variable is categorical.

        Returns
        -------
        data: numpy n x m
            The normalized data.
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if maxv[v] != minv[v]:
            data[:, v] = (data[:, v] - minv[v])/(maxv[v] - minv[v])
    return data


def standardize_data(data, mean, std):
    """
        Standardizes the data given the mean and standard deviations values of
        each variable.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        mean, std: numpy m
            The mean and standard deviation of the variable. Zero and one, resp.
            if the variable is categorical.

        Returns
        -------
        data: numpy n x m
            The standardized data.
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if std[v] > 0:
            data[:, v] = (data[:, v] - mean[v])/(std[v])
            #  Clip values more than 6 standard deviations from the mean
            data[:, v] = np.clip(data[:, v], -6, 6)
    return data


def is_continuous(data):
    """
        Returns true if data was sampled from a continuous variables, and false
        Otherwise.

        Parameters
        ----------
        data: numpy
            One dimensional array containing the values of one variable.
    """
    observed = data[~np.isnan(data)]  # not consider missing values for this.
    rules = [np.min(observed) < 0,
             np.sum((observed) != np.round(observed)) > 0,
             len(np.unique(observed)) > min(30, len(observed)/3)]
    if any(rules):
        return True
    else:
        return False


@njit
def depfunc(i, deplist):
    """ Auxiliary function to assign clusters. See get_indep_clusters. """
    if deplist[i] == i:
        return i
    else:
        return deplist[i]


@njit
def get_indep_clusters(data, scope, ncat, thr):
    """
        Cluster the variables in data into independent clusters.

        Parameters
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        scope: list
            The column indices of the variables to consider.
        ncat: numpy m
            The number of categories of each variable. One if its continuous.
        thr: float
            The threshold (p-value) below which we reject the hypothesis of
            independence. In that case, they are considered dependent and
            assigned to the same cluster.

        Returns
        -------
        clu: numpy m
            The cluster assigned to each variable.
            clu[m] is the cluster to which the mth variable is assigned.

    """
    n = len(scope)  # number of variables in the scope
    # Dependence list assuming all variables are independent.
    deplist = [i for i in range(n)]

    # Loop through all variables computing pairwise independence tests
    for i in range(0, (n-1)):
        for j in range(i+1, n):
            fatheri = depfunc(i, deplist)
            deplist[i] = fatheri
            fatherj = depfunc(j, deplist)
            deplist[j] = fatherj
            if fatheri != fatherj:
                v = 1
                unii = len(np.unique(data[~np.isnan(data[:, scope[i]]), scope[i]]))
                unij = len(np.unique(data[~np.isnan(data[:, scope[j]]), scope[j]]))
                mask = (~np.isnan(data[:, scope[j]]) * ~np.isnan(data[:, scope[i]]))
                m = np.sum(mask)
                if unii > 1 and unij > 1:
                    vari, varj = data[mask, scope[i]], data[mask , scope[j]]
                    if m > 4 and ncat[scope[i]] == 1 and ncat[scope[j]] == 1:
                        # both continuous
                        _, v = kendalltau(vari, varj)
                    elif m > 4*unij and ncat[scope[i]] == 1 and ncat[scope[j]] > 1:
                        # i continuous, j discrete
                        _, v = kruskal(vari, varj)
                    elif m > 4*unii and ncat[scope[i]] > 1 and ncat[scope[j]] == 0:
                        # i discrete, j continuous
                        _, v = kruskal(vari, varj)
                    elif m > unii*unij*2 and ncat[scope[i]] > 1 and ncat[scope[j]] > 1:
                        ## both discrete
                        v = chi_test(vari, varj)
                    if (v < thr) or np.isnan(v):  # not independent -> same cluster
                        deplist[fatherj] = fatheri

    clu = np.zeros(n)
    for i in range(n):
        clu[i] = depfunc(i, deplist)
    unique_clu = np.unique(clu)
    for i in range(n):
        clu[i] = np.min(np.where(clu[i] == unique_clu)[0])
    return clu


@njit
def gini(counts):
    """
        Computes the gini score in the distribution of classes.
        The higher the gini score, the 'purer' the distribution.
    """
    p = counts/np.sum(counts)
    return np.sum(p*(p-1))

@njit
def entropy(counts):
    """
        Computes the entropy in the distribution of classes.
        Returns -1*entropy as we want to maximize instead of minimize the score.
    """
    p = counts/np.sum(counts) + 1e-12
    return np.sum(p*np.log2(p))


@njit
def purity(counts):
    """
        Purity is a measure defined by the ratio between the number of counts
        of the majority class over the total number of instances.
        The larger the purity, the better the cluster.
    """
    if counts.sum() == 0:
        return 1
    return np.max(counts)/np.sum(counts)


@njit
def gain(ratio, left_counts, right_counts, imp):
    """
        Computes the gain of a split.

        Parameters
        ----------
        y: numpy
            The class instances in the original node.
        y_left, y_right: numpy
            The class instances in the (candidates) left and right nodes.
        imp: callable
            The impurity function. Currently, either gini, purity or entropy.
    """
    return ratio*imp(left_counts) + (1-ratio)*imp(right_counts)


@njit
def all_splits(s):
    """
        all_splits([1,2,3,4]) --> (1,) (2,) (1,2,) (3,) (1,3) (2,3) (1,2,3)
        Note that splits with an empty complementing set are not considered.
        The input is assumed to be a numpy array.
    """
    x = s.shape[0]
    for i in range(1, int((1 << x)/2)):
        yield(np.array([s[j] for j in range(x) if (i & (1 << j))]))


@njit(parallel=True)
def categorical_split(x, y, imp_measure, min_samples_leaf, n_classes):
    """
        Looks for the best split for categorical variables. It differs from
        a numerical split in that it is not sufficient to divide values above
        and below are threshold because the variable is not necessarily ordinal.
        Because of that all possible subsets of the categories need to be
        considered for a split.

        Parameters
        ----------
        x: numpy
            One dimensional array containing the realisations of a categorical
            variable we want to split on.
        y: numpy
            The class variable corresponding to the observations in x.
        imp_measure: str
            The name of the impurity measure to use.
            Currently, either gini, purity or entropy.

        Returns
        -------
        best_score: float
            The score of the best split.
        best_split: list
            List containing the categories on one side of the split. The
            categories on the other side are not returned but are trivially
            the complement of best_split.
    """
    assert x.shape[0] == y.shape[0], "Inputs should have same dimension."
    categories = np.unique(x[~np.isnan(x)])
    if categories.shape[0] < 2:  # no splits possible here
        return -np.Inf, np.array([-np.Inf])
    n_splits = int(2**(categories.shape[0]-1))
    scores = np.zeros(n_splits-1) -np.Inf

    total_counts = bincount(y, n_classes)
    for i in range(1, n_splits):
        split = np.array([categories[j] for j in range(categories.shape[0]) if (i & (1 << j))])
        y_left = y[isin_nb(x, split)]
        left_counts = bincount(y_left, n_classes)
        right_counts = total_counts - left_counts
        ratio = y_left.size/y.size
        if (y_left.size < min_samples_leaf) or (y.size - y_left.size < min_samples_leaf):
            scores[i-1] = -np.Inf
        elif imp_measure == 'entropy':
            scores[i-1] = gain(ratio, left_counts, right_counts, entropy)
        elif imp_measure == 'purity':
            scores[i-1] = gain(ratio, left_counts, right_counts, purity)
        elif imp_measure == 'gini':
            scores[i-1] = gain(ratio, left_counts, right_counts, gini)
    best_score, best_i = np.max(scores), np.argmax(scores)+1
    best_split = np.array([categories[j] for j in range(categories.shape[0]) if (best_i & (1 << j))])
    return best_score, best_split


@njit(parallel=True)
def numerical_split(x, y, imp_measure, min_samples_leaf, n_classes):
    """
        Looks for the best split for numerical variables.

        Parameters
        ----------
        x: numpy
            One dimensional array containing the realisations of a categorical
            variable we want to split on.
        y: numpy
            The class variable corresponding to the observations in x.
        imp_measure: str
            The name of the impurity measure to use.
            Currently, either gini, purity or entropy.

        Returns
        -------
        best_score: float
            The score of the best split.
        best_split: float
            The threshold on each variables are split.
    """
    assert x.shape[0] == y.shape[0], "Inputs should have same dimension."
    order = np.argsort(x)
    values = np.sort(np.unique(x))
    y_ord = y[order]
    if y.size <= 1 or values.shape[0] == 1:
        return -np.Inf, -np.Inf
    values = (values[1:] + values[:-1]) / 2
    scores = np.zeros(values.size) -np.Inf

    total_counts = bincount(y, n_classes)
    idx = np.searchsorted(x[order], values)
    for i in prange(idx.size):
        if (idx[i] > min_samples_leaf) and (y.size-idx[i] > min_samples_leaf):
            left_counts = bincount(y_ord[:idx[i]], n_classes)
            right_counts = total_counts - left_counts
            ratio = idx[i]/y.size
            if imp_measure == 'gini':
                scores[i] = gain(ratio, left_counts, right_counts, gini)
            if imp_measure == 'purity':
                scores[i] = gain(ratio, left_counts, right_counts, purity)
            if imp_measure == 'entropy':
                scores[i] = gain(ratio, left_counts, right_counts, entropy)
    best_score = np.max(scores)
    best_split = values[np.argmax(scores)]
    return best_score, best_split


@njit
def numerical_split_birigui(x, y, total_counts, imp_measure='gini', min_samples_leaf=1, n_classes=2):
    order = np.argsort(x)
    x_ord = x[order]
    y_ord = y[order]
    values = np.unique(x_ord)
    if y.size <= 1 or values.shape[0] == 1:
        return -np.Inf, -np.Inf, x_ord, y_ord, None, None
    values = (values[1:] + values[:-1]) / 2
    scores = np.zeros(values.size) -np.Inf

    total_counts = bincount(y, n_classes)
    idx = np.searchsorted(x[order], values)
    for i in prange(idx.size):
        if (idx[i] > min_samples_leaf) and (y.size-idx[i] > min_samples_leaf):
            left_counts = bincount(y_ord[:idx[i]], n_classes)
            right_counts = total_counts - left_counts
            ratio = idx[i]/y.size
            if imp_measure == 'gini':
                scores[i] = gain(ratio, left_counts, right_counts, gini)
            if imp_measure == 'purity':
                scores[i] = gain(ratio, left_counts, right_counts, purity)
            if imp_measure == 'entropy':
                scores[i] = gain(ratio, left_counts, right_counts, entropy)
    best_i = np.argmax(scores)
    best_score = scores[best_i]
    best_split = values[best_i]
    x_left, x_right = x_ord[:best_i], x_ord[best_i:]
    y_left, y_right = y_ord[:best_i], y_ord[best_i:]
    return best_score, best_split, x_left, y_left, x_right, y_right


def find_best_split(X, y, ncat, imp_measure, max_features, min_samples_leaf):
    """
        Looks for the best split among all the variables in X.
        Parameters
        ----------
        x: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        y: numpy n
            The class variable corresponding to the observations in X.
        ncat: numpy m
            The number of categories in each variable. One for numerical variables.
            The class variable is assumed to be the last one.
        imp_measure: str
            The name of the impurity measure to use.
            Currently, either gini, purity or entropy.
        max_features: int
            The maximum number of variables to consider for a split. If set to
            'auto', one third of the variables is considered, i.e. m/3.
        Returns
        -------
        split_var: float
            The best variable to split on.
        split_value: float (numerical variables) or list (categorical variables)
            Threshold that defines the best split on split_var.
    """
    best_score = -1e12
    split_var = None
    split_value = None
    vars = np.random.choice(np.arange(X.shape[1]), max_features, replace=False)
    for var in vars:
        split_X = X[~np.isnan(X[:, var]), :]
        split_y = y[~np.isnan(X[:, var])]
        if ncat[var] > 1:
            score, value = categorical_split(split_X[:, var], split_y, imp_measure, min_samples_leaf, int(ncat[-1]))
        else:
            score, value = numerical_split(split_X[:, var], split_y, imp_measure, min_samples_leaf, int(ncat[-1]))
        if score > best_score:
            best_score = score
            split_var = var
            split_value = value
    return split_var, split_value


def scope2string(scope):
    """
        Auxiliary function that converts the scope (list of ints) into a string
        for printing.
    """
    if len(scope) <= 5:
        return scope
    res = ''
    first = scope[0]
    last = None
    for i in range(1, len(scope)-1):
        if scope[i-1] != scope[i]-1:
            first = scope[i]
        if scope[i]+1 != scope[i+1]:
            last = scope[i]
        if first is not None and last is not None:
            res += str(first) + ':' + str(last) + ' - '
            first = None
            last = None
    res += str(first) + ':' + str(scope[-1])
    return res


def get_counts(node_data, n_classes):
    """
        Returns the class counts in the data.

        Parameters
        ----------
        node_data: numpy array
            The data reaching the node.
        n_classes: int
            Number of classes in the data. This is needed as not
            every class is necessarily observed in node_data.

        Returns
        -------
        counts: numpy array (same dimension as all_classes)
            The class counts in node_data.
    """
    return bincount(node_data, n_classes)


class Dist:
    """
        Class that defines a (truncated) Gaussian density function.

        Attributes
        ----------
        scope: list of ints
            The variables over which the Gaussian is defined.
        lower: float
            The minimum value the variable might assume.
            Currently, only applicable for univariate Gaussians.
        upper: float
            The maximum value the variable might assume.
            Currently, only applicable for univariate Gaussians.
        n: int
            The number of data points used to fit the Gaussian.
        mean: float
            The empirical mean.
        cov: float
            The empirical variance, covariance
    """
    def __init__(self, scope, data=None, n=None, lower=-np.Inf, upper=np.Inf):
        if not isinstance(scope, list):
            scope = [scope]
        self.scope = scope
        self.lower = lower
        self.upper = upper
        if data is not None:
            self.n = data.shape[0]
            self.fit(data)
        else:
            self.n = n
            self.mean = None
            self.cov = None

    def fit(self, data):
        """
            Fits mean and convariance using data. Also normalizes the upper and
            lower thresholds to define where to truncate the Gaussian.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        self.n = data.shape[0]
        assert self.n > 0, "Empty data"
        self.mean = np.nanmean(data[:, self.scope], axis=0)
        # assert ~np.isnan(self.mean), "Error computing the mean."
        if data[:, self.scope].shape[0] > 1:  # Avoid runtimewarning
            self.cov = np.cov(data[:, self.scope], rowvar=False)
            self.cov = np.where(self.cov > .1, self.cov, .1)
        else:
            self.cov = np.array(.1)  # Probably not the best solution here
        self.std = np.sqrt(self.cov)

        # Compute the tresholds to truncate the Gaussian.
        # The Gaussian has support [a, b]
        self.a = (self.lower - self.mean) / self.std
        self.b = (self.upper - self.mean) / self.std
        self.params = {'loc': self.mean, 'scale':self.std, 'a': self.a, 'b': self.b}
        return self

    def logpdf(self, data):
        """
            Computes the log-density at data.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        complete_ind = np.where(~np.isnan(data[:, self.scope]).any(axis=1))[0]
        # Initialize log-density to zero (default value for non-observed variables)
        logprs = np.zeros(data.shape[0])
        logprs[complete_ind] = stats.truncnorm.logpdf(data[complete_ind, self.scope], **self.params).reshape(-1)
        return logprs.reshape(data.shape[0], 1)

    def logpdf_one(self, data):
        """
            Computes the log-density at data.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        if np.isnan(data[self.scope]):
            return 0
        else:
            return stats.truncnorm.logpdf(data[self.scope], **self.params)[0]
        return logprs

    def max_logpdf(self, data):
        """
            Runs max inference at data.

            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
        """
        complete_ind = np.where(~np.isnan(data[:, self.scope]).any(axis=1))[0]
        incomplete_ind = np.where(np.isnan(data[:, self.scope]).any(axis=1))[0]
        # Initialize log-density to zero (default value for non-observed variables)
        logprs = np.zeros(data.shape[0])
        logprs[complete_ind] = stats.truncnorm.logpdf(data[complete_ind, self.scope], **self.params).reshape(-1)
        logprs[incomplete_ind] = self.mean
        return logprs


#  Clustering utils
def make_cost_matrix(c1, c2):
    """
    Computes the cost matrix used for the Munkres (Hungarian) algorithm.
    Each element m[i, j] is the number of instances where c1 = i and c2 = j
    multiplied by -1, which represents the cost of matching labels i and j.
    Intuitively, the larger m[i, j] the better the match and lower the cost,
    hence the change of sign above.

    Parameters
    ----------

    c1, c2: numpy array
        The clustering (or classification) assignements to be matched.
    """
    uc1 = np.unique(c1)  # Unique labels
    uc2 = np.unique(c2)
    l1 = uc1.size  # Number of labels
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2))  # We assume labels are the same

    m = np.ones([l1, l2])  # Create template matrix
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m


def munkres_accuracy(true_labels, pred_labels):
    """
    Computes how well the predicted labels match the true ones.
    The Munkres algorithm finds the best matching between the two label
    assignements, so the actual labels do not matter.

    Parameters
    ----------
    true_labels: numpy array
        The true or previous label of each instance.
    pred_labels: numpy array
        The predicted label of each instance.

    """
    assert true_labels.shape[0] == pred_labels.shape[0], "The label vectors must have the same size."
    num_labels = true_labels.shape[0]
    cost_matrix = make_cost_matrix(pred_labels, true_labels)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = { old: new for (old, new) in indexes }
    new_labels = translate_clustering(pred_labels, mapper)
    new_cm = confusion_matrix(true_labels, new_labels, labels=range(num_labels))
    return accuracy(new_cm)


def translate_clustering(clt, mapper):
    """ Maps a vector labels (clt) to another according to a map (mapper). """
    return np.array([ mapper[i] for i in clt ])


def accuracy(cm):
    """
    Computes accuracy from confusion matrix.

    Parameters
    ----------
    cm: 2D numpy array
        Confusion matrix.
    """
    return np.trace(cm, dtype=float) / np.sum(cm)
