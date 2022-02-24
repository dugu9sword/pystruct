import numpy as np

from .crf import CRF
from ..utils import expand_sym, compress_sym


class GraphCRF(CRF):
    """Pairwise CRF on a general graph.

    Pairwise potentials the same for all edges, are symmetric by default
    (``directed=False``).  This leads to n_classes parameters for unary
    potentials.

    If ``directed=True``, there are ``n_classes * n_classes`` parameters
    for pairwise potentials, if ``directed=False``, there are only
    ``n_classes * (n_classes + 1) / 2`` (for a symmetric matrix).

    Examples, i.e. X, are given as an iterable of n_examples.
    An example, x, is represented as a tuple (features, edges) where
    features is a numpy array of shape (n_nodes, n_attributes), and
    edges is is an array of shape (n_edges, 2), representing the graph.

    Labels, Y, are given as an iterable of n_examples. Each label, y, in Y
    is given by a numpy array of shape (n_nodes,).

    There are n_states * n_features parameters for unary
    potentials. For edge potential parameters, there are n_state *
    n_states permutations, i.e. ::

                state_1 state_2 state 3
        state_1       1       2       3
        state_2       4       5       6
        state_3       7       8       9

    The fitted parameters of this model will be returned as an array
    with the first n_states * n_features elements representing the
    unary potentials parameters, followed by the edge potential
    parameters.

    Say we have two state, A and B, and two features 1 and 2. The unary
    potential parameters will be returned as [A1, A2, B1, B2].

    If ``directed=True`` the edge potential parameters will return
    n_states * n_states parameters. The rows are senders and the
    columns are recievers, i.e. the edge potential state_2 -> state_1
    is [2,1]; 4 in the above matrix.

    The above edge potential parameters example would be returned as
    [1, 2, 3, 4, 5, 6, 7, 8, 9] (see numpy.ravel).

    If edges are undirected, the edge potential parameter matrix is
    assumed to be symmetric and only the lower triangle is returned, i.e.
    [1, 4, 5, 7, 8, 9].


    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables. Inferred from data if not provided.

    n_features : int, default=None
        Number of features per node. Inferred from data if not provided.

    inference_method : string or None, default=None
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

        If None, ad3 is used if installed, otherwise lp.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    directed : boolean, default=False
        Whether to model directed or undirected connections.
        In undirected models, interaction terms are symmetric,
        so an edge ``a -> b`` has the same energy as ``b -> a``.

    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None, directed=False):
        self.directed = directed
        CRF.__init__(self, n_states, n_features, inference_method,
                     class_weight=class_weight)
        # n_states unary parameters, upper triangular for pairwise

    def _set_size_joint_feature(self):
        # try to set the size of joint_feature if possible
        if self.n_features is not None and self.n_states is not None:
            if self.directed:
                self.size_joint_feature = int(self.n_states * self.n_features +
                                           self.n_states ** 2)
            else:
                self.size_joint_feature = int(
                    self.n_states * self.n_features
                    + self.n_states * (self.n_states + 1) / 2)

    def _get_edges(self, x):
        return x[1]

    def _get_features(self, x):
        return x[0]

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        pw = w[self.n_states * self.n_features:]
        if self.directed:
            return pw.reshape(self.n_states, self.n_states)
        return expand_sym(pw)

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features = self._get_features(x)
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)

        return np.dot(features, unary_params.T)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        @comment: Yi
        x is a tuple:
            - node's feature:   ndarray(n_nodes, n_features)
                by convention, we hope the feature space to be discrete, which means
                does a feature appears or how many times does it appear.
                actually we can allow any dense feature without loss of generality.
            - nodes' structure: ndarray(n_edges, 2)
        y is:
            - node's marginal:  ndarray(n_nodes, n_states)
                it can be discrete or continous (but sum to 1), so we call it 
                marginal since it is a proability in general.
        
        Note that (x, y)'s shape is different case-by-case since they have different
        number of nodes and edges. We try to convert it into a fix-sized dense vector 
        so that it can be directly multiplied with the weight vector.

        Here is an example: Denote the feature dim as F, the state space dim as S. Considering
        a instance where we have N nodes and E edges. Supposing the marginal matrix is M{NS} 
        (which means M's dim is NxS), the feature matrix is F{NF}.

        the unary potential should be:
            (\sum_n M{SN} @ F{NF}){SF} ele-wise-dot W{SF}
        
        WHY? considering a single-node discrete case, the first vector denotes the 1-st 
        state occurs, the second vector denotes the 0-th feature occures. A outer product 
        can deduce the co-occurence of states and features.

            [0]             [0 0 0]
            [1] @ [1 0 0] = [1 0 0]
            [0]             [0 0 0]

        Parameters
        ----------
        x : tuple
            Unary evidence.

        y : ndarray or tuple
            Either y is an integral ndarray, giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            """
                get accumulated pairwise feature count table (in |state|^2 space) 

                unary_marginal:   n_nodes x n_state
                    -> one-hot, the i-th node has the j-th state
                unary_marginals[edges[:, 0]]:   n_edge x n_state
                    -> one hot, the i-th edge.src has the j-th state
                unary_marginals[edges[:, 1]]:   n_edge x n_state
                    -> one hot, the i-th edge.tgt has the j-th state
                pw:   n_state x n_state
                    -> count of (i-th state -> j-th state) in edges
            """
            y = y.reshape(n_nodes)
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            unary_marginals[np.ogrid[:n_nodes], y] = 1

            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])

        """
            get accumulated unary feature table (in |state| x |feature| space)
        """
        unaries_acc = np.dot(unary_marginals.T, features)   
        if self.directed:
            pw = pw.ravel()
        else:
            pw = compress_sym(pw)

        joint_feature_vector = np.hstack([unaries_acc.ravel(), pw])
        return joint_feature_vector
