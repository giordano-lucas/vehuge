import numpy as np

class Node:
    def __init__(children, scope, weight, n, type, len, size, id, value=None):
        self.children = children
        self.scope = scope
        self.weight = weight
        self.n = n  ## TODO: find out what this is.
        self.type = type
        if value: self.value = value
        self.len = len
        self.size = size
        self.id = id  # Random identification number
        self.memory = dict()

class SPN:
    def __init__():
        self.root = Node()
        self.ncat = 2  # There must be at least two categories
        self.maxv = 1  # Maximum value a variable can assume
        self.minv = 0  # Minimum value a variable can assume

    def learn(self, data, max_height):
        scope = np.arrange(data.shape[1])  # Range over all variables (data cols)
        root = learn_aux

    def learn_aux(self, data, ncat, scope, thr, nclusters, verb, max_height, classcol, classvalues, last_prod=False, class_unif=False):
        n = len(scope)
        m = data.shape[0]
        if n > 1:
            pass
        else:  # Single variable in the scope
            if ncat[scope] > 1:
                ncategory = ncat[scope]
                sum_node = Node(children=[], scope=scope, weight=np.zeros(ncategory), n=m, type=4, len=ncategory, size=ncategory+1, id=np.random.randint(0, 10000000))
