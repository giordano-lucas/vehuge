from functools import reduce
import numpy as np
import operator
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from learning import LearnSPN
from nb_nodes import SumNode, ProdNode, Leaf, GaussianLeaf, MultinomialLeaf
from nb_nodes import fit_indicator, fit_gaussian, fit_multinomial, fit_multinomial_with_counts
from spn import SPN
from utils import is_continuous, learncats


def tree2spn(tree, X, y, learnspn=False, uniformleaves=False, uni_everywhere=False):
    """
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    """
    feature_names = [i for i in range(X.shape[1])]
    if hasattr(tree, 'tree_'):
        tree_ = tree.tree_
    else:
        tree_ = tree
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    scope = np.array([i for i in range(X.shape[1]+1)]).astype(int)
    data = np.concatenate([X, np.expand_dims(y, axis=1)], axis=1)
    ncat = np.array(tree.ncat, dtype=int)
    lp = np.sum(np.where(ncat==1, 0, ncat)) * 1e-6 # LaPlace counts
    upper = ncat.copy().astype(float)
    upper[upper == 1] = np.Inf
    lower = ncat.copy().astype(float)
    lower[ncat == 1] = -np.Inf
    lower = np.ones(data.shape[1])*(-np.Inf)
    classcol = len(ncat)-1

    def recurse(node, node_ind, depth, data, upper, lower):
        indent = "  " * depth
        value = tree_.value[node_ind][0]
        counts = np.bincount(data[:, -1].astype(int), minlength=int(ncat[-1]))
        # assert all(value==counts+1e-6), (value, counts, node_ind)

        if tree_.feature[node_ind] != _tree.TREE_UNDEFINED:
            split_col = feature_name[node_ind]
            split_value = tree_.threshold[node_ind]

            sumnode = SumNode(scope=scope, parent=node, n=data.shape[0]+lp)

            split_value = np.array([split_value]).astype(float)
            upper1 = upper.copy()
            lower1 = lower.copy()
            upper1[split_col] = min(split_value, upper1[split_col])
            split1 = data[np.where(data[:, split_col] <= split_value)]
            p1 = ProdNode(scope=scope, parent=sumnode, n=split1.shape[0]+lp)
            ind1 = Leaf(scope=np.array([split_col]), parent=p1, n=split1.shape[0]+lp, value=split_value, comparison=3)  # Comparison <=
            recurse(p1, tree_.children_left[node_ind], depth + 1, split1.copy(), upper1, lower1)

            upper2 = upper.copy()
            lower2 = lower.copy()
            lower2[split_col] = max(split_value, lower2[split_col])
            split2 = data[np.where(data[:, split_col] > split_value)]
            p2 = ProdNode(scope=scope, parent=sumnode, n=split2.shape[0]+lp)
            ind2 = Leaf(scope=np.array([split_col]), parent=p2, n=split2.shape[0]+lp, value=split_value, comparison=4)  # Comparison >
            recurse(p2, tree_.children_right[node_ind], depth + 1, split2.copy(), upper2, lower2)

            return sumnode
        else:
            if (learnspn) and (data.shape[0] >= 30):
                # We learn SPN on the explanatory variables X
                prodnode = ProdNode(scope=scope, parent=node, n=data.shape[0]+lp)
                # The class variable is modeled independently as multinomial data
                leaf = MultinomialLeaf(scope=np.array([data.shape[1]-1]), parent=prodnode, n=data.shape[0]+lp)
                fit_multinomial(leaf, data, int(ncat[data.shape[1]-1]))
                # Other variables are modelled with a LearnSPN
                learner = LearnSPN(ncat=tree.ncat[:-1], classcol=None)
                learner.fit(data[:, :-1], last_node=prodnode)
            elif (uniformleaves in ['const', 'per_leaf']) and ((data.shape[0] < 10) or uni_everywhere):
                classleaf = MultinomialLeaf(scope=np.array([scope[-1]]), parent=node, n=data.shape[0]+lp)
                fit_multinomial(classleaf, data, int(ncat[data.shape[1]-1]))
                # Uniform distribution defined on [-3, 3]
                # Assuming mean 0 and std 1, this should cover most of the data
                if uniformleaves == 'per_leaf':
                    vol = np.array([min(upper[i] - lower[i], 6) if ncat[i]==1 else upper[i] for i in range(data.shape[1]-1)])
                else:
                    vol = np.array([6 if nc==1 else nc for nc in ncat])
                eps = np.log([reduce(operator.mul, 1/vol)]).astype(float)
                uniformleaf = UniformLeaf(scope=scope[:-1], parent=node, n=data.shape[0]+lp, value=eps)
                return None
            else:
                for var in scope:
                    if ncat[var] > 1:
                        if var == classcol:
                            leaf = MultinomialLeaf(scope=np.array([var]), parent=node, n=data.shape[0]+lp)
                            fit_multinomial_with_counts(leaf, value)
                        else:
                            leaf = MultinomialLeaf(scope=np.array([var]), parent=node, n=data.shape[0]+lp)
                            fit_multinomial(leaf, data, int(ncat[var]))
                    else:
                        leaf = GaussianLeaf(scope=np.array([var]), parent=node, n=data.shape[0]+lp)
                        fit_gaussian(leaf, data, upper[var], lower[var])
                return None

    spn = SPN()
    spn.root = recurse(None, 0, 1, data, upper, lower)
    spn.ncat = ncat

    return spn

class SKRandomForest:
    def __init__(self, n_estimators=100, max_features='auto', bootstrap=True, oob_score=False,
                 n_jobs=-1, verbose=False, warm_start=False, ccp_alpha=0.0,
                 max_samples=None, ncat=None, **kwargs):

        self.n_estimators = n_estimators
        self.estimators = []
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.ncat = ncat
        self.estimator_params = kwargs
        self.estimator_params['max_features'] = max_features

    def fit(self, X, y):
        n = X.shape[0]
        self.scope = np.array([i for i in range(X.shape[1]+1)]).astype(int)
        self.classes = np.unique(y)
        for i in range(self.n_estimators):
            if self.bootstrap:
                repeat = True
                while repeat:  # needed to avoid samples without all classes
                    shuffle = np.random.choice(np.arange(n), n, replace=True)
                    X_tree = X[shuffle, :]
                    y_tree = y[shuffle]
                    if set(np.unique(y_tree)) == set(self.classes):
                        repeat = False
            tree = DecisionTreeClassifier(**self.estimator_params)
            tree.fit(X_tree, y_tree)
            tree.ncat = self.ncat # Saving data info with the tree
            # Add LaPlace smoothing
            for v in tree.tree_.value:
                v += 1e-6
            self.estimators.append({'tree': tree, 'X': X_tree, 'y': y_tree})

    def tospn(self, learnspn=False, uniformleaves=False):
        """ Returns an SPN Model. """
        spn = SPN()
        spn.root = SumNode(scope=self.scope, parent=None, n=1)
        for estimator in self.estimators:
            tree_spn = tree2spn(**estimator, learnspn=learnspn, uniformleaves=uniformleaves)
            spn.root.add_child(tree_spn.root)
        spn.ncat = tree_spn.ncat
        return spn

    def torf(self):
        """ Returns an SKlearn Random Forest Model. """
        rf = RandomForestClassifier(n_estimators=self.n_estimators)
        estimators = [est['tree'] for est in self.estimators]
        rf.estimators_ = estimators
        # Fetch sklearn parameters from one of the tree estimators
        rf.classes_ = estimators[0].classes_
        rf.n_classes_ = estimators[0].n_classes_
        rf.n_outputs_ = estimators[0].n_outputs_
        return rf

    def predict(self, X, vote=True):
        if vote:
            votes = np.zeros(shape=(X.shape[0], self.n_estimators))
            for i, estimator in enumerate(self.estimators):
                tree = estimator['tree']
                votes[:, i] = tree.predict(X)
            return stats.mode(votes, axis=1)[0].reshape(-1)
        else:
            probas = np.zeros(shape=(X.shape[0], self.n_estimators, self.classes.shape[0]))
            for i, estimator in enumerate(self.estimators):
                tree = estimator['tree']
                probas[:, i] = tree.predict_proba(X)
            return np.mean(probas, axis=1)
