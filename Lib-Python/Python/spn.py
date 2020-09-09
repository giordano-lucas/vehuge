import numpy as np
import scipy.stats as stats

from learning import LearnSPN
from nb_nodes import SumNode, ProdNode, Leaf, GaussianLeaf, eval_root, eval_root_children, eval_root_class, delete
from utils import chi_test, learncats, get_stats, normalize_data, logsumexp3


class SPN:
    """
        Class that defines and evaluates an SPN.

        Attributes
        ----------

        root: Node object
            The root node of the SPN.
        ncat: numpy
            The number of categories of each variable. One for continuous variables.
        learner: object
            Defines the learning method of the SPN.
            Currently, only LearnSPN (Gens and Domingos, 2013).
        nparams: int
            The number of parameters in the SPN.
    """

    def __init__(self):
        self.root = None
        self.maxv = None
        self.minv = None
        self.n_nodes = 0

    @property
    def nparams(self):
        return self.root.nparams

    def update(self, node):
        if node['id'] >= len(self.nodes):
            print('Increasing')
            self.nodes = np.concatenate([self.nodes, np.zeros(100, dtype=NODE_DTYPE)])
        self.nodes[node['id']] = node
        self.n_nodes += 1

    def learnSPN(self, data, classcol=None, thr=0.001, nclusters=2, max_height=1e8):
        """
            Learns an SPN using the LearnSPN algorithm.

            Parameters
            ----------

            data: numpy array
                Instances as rows; variables as columns.
            classcol: int
                The index of the column corresponding to the class variable.
            thr: float
                p-value threshold for independence tests in product nodes.
            nclustes: int
                Number of clusters in sum nodes.
            max_height: int
                Maximum height (depth) of the network.
        """

        self.ncat = learncats(data, classcol)
        data, self.maxv, self.minv, _, _ = get_stats(data)
        self.learner = LearnSPN(self.ncat, thr, nclusters, max_height, classcol)
        self.root = self.learner.fit(data)
        return self

    def ensemble_learnSPN(self, data, n_estimators, classcol=None,
                 thr=0.001, nclusters=2, max_height=1000000, max_features='auto'):
        """

            Parameters
            ----------

            data: numpy array
                Instances as rows; variables as columns.
            n_estimators: int
                Number of models in the ensemble. Each one corresponds to a
                different run of LearnSPN.
            classcol: int
                The index of the column corresponding to the class variable.
            thr: float
                p-value threshold for independence tests in product nodes.
            nclustes: int
                Number of clusters in sum nodes.
            max_height: int
                Maximum height (depth) of the network.
            max_features: int or 'auto'
                Maximum number of features to consider at each split.
                Akin to what is done in a random forest.
                Default is 'auto', which means one third of the features.
        """

        self.ncat = learncats(data, classcol)
        data, self.maxv, self.minv, _, _ = get_stats(data)
        self.learner = LearnSPN(self.ncat, thr, nclusters, max_height, classcol, max_features)
        scope = [i for i in range(data.shape[1])]
        self.root = SumNode(scope=scope, parent=None, n=data.shape[0])
        n = data.shape[0]
        if classcol is not None:
            n_classes = len(np.unique(data[:, classcol]))
        for i in range(n_estimators):
            repeat = True
            while repeat:  # needed to avoid samples without all classes
                shuffle = np.random.choice(np.arange(n), n, replace=True)
                data_tree = data[shuffle, :]
                if (classcol is None) or (len(np.unique(data_tree[:, classcol])) == n_classes):
                    repeat = False
            self.root.add_child(self.learner.fit(data_tree))
        return self

    def set_topological_order(self):
        """
            Updates the ids of the nodes so that they match their topological
            order.
        """

        def get_topological_order(node, order=[]):
            if order == []:
                node.id = len(order)
                order.append(node)
            for child in node.children:
                child.id = len(order)
                order.append(child)
            for child in node.children:
                get_topological_order(child, order)
            return order
        self.order = get_topological_order(self.root, [])
        self.n_nodes = len(self.order)

    def log_likelihood(self, data):
        """
            Computes the log-likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        return eval_root(self.root, data)

    def likelihood(self, data):
        """
            Computes the likelihood of data.

            Parameters
            ----------

            data: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
        """

        ll = self.log_likelihood(data)
        return np.exp(ll)

    def classify(self, X, classcol, return_prob=False, norm=False):
        """
            Classifies instances running proper SPN inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            norm: boolean (dafault False)
                Whether to normalize the data before running inference.
        """

        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if norm and self.maxv is not None:
            X = normalize_data(X, self.maxv, self.minv)
        nclass = int(self.ncat[classcol])
        joints = eval_root_class(self.root, X, classcol, nclass, naive=False)
        joints = logsumexp3(joints, axis=2)
        joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        probs = probs/probs.sum(axis=1, keepdims=True)
        if return_prob:
            return np.argmax(probs, axis=1), probs
        return np.argmax(probs, axis=1)

    def classify_avg(self, X, classcol, return_prob=False, norm=False, naive=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each SPN, that is,
            argmax_y sum_n P_n(Y=y|X)/n

            This is only makes sense if the SPN was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            norm: boolean (dafault False)
                Whether to normalize the data before running inference.
            naive: boolean
                Whether to treat missing values as suggested in Friedman1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
            """

        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if norm and self.maxv is not None:
            X = normalize_data(X, self.maxv, self.minv)
        nclass = int(self.ncat[classcol])
        # joints = np.zeros((X.shape[0], nclass, self.root.nchildren))
        joints = eval_root_class(self.root, X, classcol, nclass, naive)
        if naive:
            counts = np.exp(joints).astype(int)  # int to filter out the smoothing
            conditional = counts/np.sum(counts, axis=1, keepdims=True)
        else:
            # Convert from log to probability space
            joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
            probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
            # Normalize to sum out X: we get P(Y|X) by dividing by P(X)
            conditional = probs/np.sum(probs, axis=1, keepdims=True)
        # Average over the trees
        agg = np.mean(conditional, axis=2)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass

    def classify_lspn(self, X, classcol, return_prob=False, norm=False):
        """
            Classifies instances running proper SPN inference, that is,
            argmax_y P(X, Y=y).

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            norm: boolean (dafault False)
                Whether to normalize the data before running inference.
            naive: boolean
                Whether to treat missing values as suggested in Friedman1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
        """

        eps = 1e-6
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if norm and self.maxv is not None:
            X = normalize_data(X, self.maxv, self.minv)
        nclass = int(self.ncat[classcol])
        maxclass = np.zeros(X.shape[0])-1
        maxlogpr = np.zeros(X.shape[0])-np.Inf
        joints = np.zeros((X.shape[0], nclass))
        for i in range(nclass):
            iclass = np.zeros((X.shape[0], 1)) + i
            Xi = np.append(X, iclass, axis=1)
            joints[:, i] = np.squeeze(eval_root(self.root, Xi))
        joints_minus_max = joints - np.max(joints, axis=1, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        probs = probs/probs.sum(axis=1, keepdims=True)
        if return_prob:
            return np.argmax(probs, axis=1), probs
        return np.argmax(probs, axis=1)

    def classify_avg_lspn(self, X, classcol, return_prob=False, norm=False):
        """
            Classifies instances by taking the average of the conditional
            probabilities defined by each SPN, that is,
            argmax_y sum_n P_n(Y=y|X)/n
            This is only makes sense if the SPN was learned as an ensemble, where
            each model is the child of the root.

            Parameters
            ----------

            X: numpy array
                Input data not including the class variable.
                Missing values should be set to numpy.nan
            classcol: int
                The original index of the class variable.
            return_prob: boolean
                Whether to return the conditional probability of each class.
            norm: boolean (dafault False)
                Whether to normalize the data before running inference.
            naive: boolean
                Whether to treat missing values as suggested in Friedman1975,
                that is, by taking the argmax over the counts of all pertinent
                cells.
        """

        eps = 1e-6
        if norm and self.maxv is not None:
            X = normalize_data(X, self.maxv, self.minv)
        nclass = int(self.ncat[classcol])
        joints = np.zeros((X.shape[0], self.root.nchildren, nclass))
        for i in range(nclass):
            iclass = np.zeros((X.shape[0], 1)) + i
            Xi = np.append(X, iclass, axis=1)
            joints[:, :, i] = eval_root_children(self.root, Xi)
        joints_minus_max = joints - np.max(joints, axis=2, keepdims=True)
        probs = np.where(np.exp(joints_minus_max) >= (np.log(eps) - np.log(nclass)), np.exp(joints_minus_max), 0)
        normalized = probs/np.sum(probs, axis=2, keepdims=True)
        agg = np.mean(normalized, axis=1)
        maxclass = np.argmax(agg, axis=1)
        if return_prob:
            return maxclass, agg
        return maxclass

    def mpe(self, X, norm=False):
        """
            Runs Most Probable Explanation (MEP) inference at X.

            Parameters
            ----------
            X: numpy array
                Input data including the class variable.
                Missing values should be set to numpy.nan
            norm: boolean (dafault False)
                Whether to normalize the data before running inference.
        """

        if norm and self.maxv is not None:
            X = normalize_data(X, self.maxv, self.minv)
        res = X.copy()
        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.type == 'S':
                queue.extend(node.argmax(X))
            elif node.type == 'P':
                queue.extend(node.children)
            elif node.type in ['G', 'L']:
                if np.isnan(X[:, node.scope]).any():
                    assert np.isnan(res[:, node.scope]).any(), "MPE error."
                    res[:, node.scope] = node.argmax(X)
        return res

    def AIC(self, data):
        """ Computes the AIC score given data. """

        return -2 * np.sum(self.logprobs(data)) + 2 * self.nparams

    def BIC(self, data):
        """ Computes the BIC score given data. """

        return -2 * np.sum(self.logprobs(data)) + np.log(data.shape[0]) * self.nparams

    def clear(self):
        """ Deletes the structure of the SPN. """

        if self.root is not None:
            self.root.remove_children(*self.root.children)
            self.root = None

    def isvalid(self):
        """ Checks whether the SPN obeys decomposability and completeness. """

        return self.root.isvalid()

    def get_node(self, id):
        """ Fetchs node by its id. """

        queue = [self.root]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        print("Node %d is not part of the network.", id)

    def get_node_of_type(self, type):
        """ Fetchs all nodes of a given type. """

        queue = [self.root]
        res = []
        while queue != []:
            node = queue.pop(0)
            if node.type == type:
                res.append(node)
            if node.type not in ['L', 'G', 'U']:
                queue.extend(node.children)
        return res

    def delete(self):
        """
            Calls the delete function of the root node, which in turn deletes
            the rest of the nodes in the SPN. Given that nodes in an SPN point
            to each other, they are always referenced and never automatically
            deleted by the Python interpreter.
        """

        delete(self.root)
