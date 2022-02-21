import numpy as np
from scipy import sparse

from .common import _validate_params, compute_energy
from ..utils.graph_functions import is_forest


def edges_to_graph(edges, n_vertices=None):
    if n_vertices is None:
        n_vertices = np.max(edges) + 1
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices)).tocsr()
    return graph


def is_chain(edges, n_vertices):
    """Check if edges specify a chain and are in order."""
    return (np.all(edges[:, 0] == np.arange(0, n_vertices - 1))
            and np.all(edges[:, 1] == np.arange(1, n_vertices)))


def inference_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=30, damping=0.5, tol=1e-5, relaxed=None):
    """Max-product inference.

    In case the edges specify a tree, dynamic programming is used
    producing a result in only a single pass.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function. 
        Shape: (n_states, n_states) or (n_edges, n_states, n_states)

    edges : nd-array
        Edges of energy function.

    max_iter : int (default=10)
        Maximum number of iterations. Ignored if graph is a tree.

    damping : float (default=.5)
        Daming of messages in loopy message passing.
        Ignored if graph is a tree.

    tol : float (default=1e-5)
        Stopping tollerance for loopy message passing.
    """
    from ._viterbi import viterbi
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    if is_chain(edges=edges, n_vertices=len(unary_potentials)):
        y = viterbi(unary_potentials.astype(np.float).copy(),
                    # sad second copy b/c numpy 1.6
                    np.array(pairwise_potentials, dtype=np.float))
    elif is_forest(edges=edges, n_vertices=len(unary_potentials)):
        y = tree_max_product(unary_potentials, pairwise_potentials, edges)
    else:
        y = iterative_max_product(unary_potentials, pairwise_potentials, edges,
                                  max_iter=max_iter, damping=damping)
    return y


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    """
        Parameters
        ----------
        unary_potentials:    (n_vertices, n_states)
        pairwise_potentials: (n_edges, n_states, n_states)
        edges:               (n_edges, 2)
        
        @comment: Yi
        forward  -> outside
        backward -> inside
        See https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf
        
    """
    n_vertices, n_states = unary_potentials.shape
    parents = -np.ones(n_vertices, dtype=np.int)
    visited = np.zeros(n_vertices, dtype=np.bool)
    neighbors = [[] for i in range(n_vertices)]
    pairwise_weights = [[] for i in range(n_vertices)]
    for pw, edge in zip(pairwise_potentials, edges):
        neighbors[edge[0]].append(edge[1])
        pairwise_weights[edge[0]].append(pw)
        neighbors[edge[1]].append(edge[0])
        pairwise_weights[edge[1]].append(pw.T)

    messages_forward = np.zeros((n_vertices, n_states))
    messages_backward = np.zeros((n_vertices, n_states))
    pw_forward = np.zeros((n_vertices, n_states, n_states))
    # build a breadth first search of the tree
    """
        @comment: Yi
        Given a tree with 11 vertices and 3 states:

                3
              /   \ 
             9     5
            / \     \
           0   4     2
         / | \      / \
        8  7  6    1   10
             
        BFS: [0, 9, 8, 7, 6, 3, 4, 5, 2, 1, 10]
        Decide the root: Every node in a tree can be seen as the root node 
        of the subtree rooted at that node. Each neighbour of the root node 
        is viewed as its child.
        https://en.wikipedia.org/wiki/Tree_(data_structure)
        
        Important notations:
        neighbors & pairwise_weights:
                                            | h1-c1 h1-c2 h1-c3 |
                pairwise_weights[h][c]  =   | h2-c1 h2-c2 h2-c3 |
                                            | h3-c1 h3-c2 h3-c3 |
        pw_forward:
            when traversing the tree, a neighbor is always a child of the node.
            pw_forward[neighbor]'s row denotes its parent.
                                | h1-c1 h1-c2 h1-c3 |
                pw_forward[c] = | h2-c1 h2-c2 h2-c3 |
                                | h3-c1 h3-c2 h3-c3 |
    """
    traversal = []
    lonely = 0
    while lonely < n_vertices:
        for i in range(lonely, n_vertices):
            if not visited[i]:
                queue = [i]
                lonely = i + 1
                visited[i] = True
                break
            lonely = n_vertices

        while queue:
            node = queue.pop(0)
            traversal.append(node)
            for pw, neighbor in zip(pairwise_weights[node], neighbors[node]):
                if not visited[neighbor]:
                    parents[neighbor] = node
                    queue.append(neighbor)
                    visited[neighbor] = True
                    pw_forward[neighbor] = pw

                elif not parents[node] == neighbor:
                    raise ValueError("Graph not a tree")
    # messages from leaves to root
    """
        @comment: Yi
        messages_backward saves the message from its descedents. 
        
        In numpy, (3, ) is not row/column vector, but just a 1-d array.
        (3, ) + (n, 3) is broadcasted as (n, 3) + (n, 3)
        
        msg_bw     +     unary    +       pw_forward
                                    | h1-c1 h1-c2 h1-c3 |
        |c1 c2 c3| + | c1 c2 c3 | + | h2-c1 h2-c2 h2-c3 | = similar to pw_forward
                                    | h3-c1 h3-c2 h3-c3 | 
    """
    for node in traversal[::-1]:
        parent = parents[node]
        if parent != -1:
            message = np.max(messages_backward[node] + unary_potentials[node] +
                             pw_forward[node], axis=1)
            message -= message.max()
            messages_backward[parent] += message
    # messages from root back to leaves
    """
        @comment: Yi
        messages_forward saves the message from its ancestor. 
        
        msg_fw     +     unary    +       pw_forward.T
                                    | c1-h1 c1-h2 c1-h3 |
        |h1 h2 h3| + | h1 h2 h3 | + | c2-h1 c2-h2 c2-h3 | = ...
                                    | c3-h1 c3-h2 c3-h3 | 
    """
    for node in traversal:
        parent = parents[node]
        if parent != -1:
            message = messages_forward[parent] + unary_potentials[parent] + pw_forward[node].T
            # leaves to root messages from other children
            message += messages_backward[parent] - np.max(messages_backward[node]
                                                          + unary_potentials[node]
                                                          + pw_forward[node], axis=1)
            message = message.max(axis=1)
            message -= message.max()
            messages_forward[node] += message

    return np.argmax(unary_potentials + messages_forward + messages_backward, axis=1)


def iterative_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=10, damping=.5, tol=1e-5, track_best=False):
    """
        Parameters
        ----------
        unary_potentials:    (n_vertices, n_states)
        pairwise_potentials: (n_edges, n_states, n_states)
        edges:               (n_edges, 2)
        
        @comment: Yi
        For an edge connecting two nodes 0 and 1, messages[e, 0] denotes the last message 
        sent from 0 to 1. 
        
        Firstly, in a new update, the new 0->1 message (messages[0]) is:
        - 0's all incoming messages except 1:
            +  all_incoming[0]
            -  messages[1] (messages[1] is used to compute message[0], and vice versa)
        - potentials:
            + 0's unary potential
            + (0, 1)'s pairwise potential
        
        Then, we need to update:
        - the 0->1 message (messages[0]) by damping (See MLAPP)
        - the incoming messages for 1 (all_incoming[1])
        
    """
    if track_best:
        best_configuration = None
        best_energy = -10000
    n_edges = len(edges)
    n_vertices, n_states = unary_potentials.shape
    messages = np.zeros((n_edges, 2, n_states))
    all_incoming = np.zeros((n_vertices, n_states))
    for i in range(max_iter):
        diff = 0
        for e, (edge, pairwise) in enumerate(zip(edges, pairwise_potentials)):
            # update message from edge[0] to edge[1]
            update = (all_incoming[edge[0]] - messages[e, 1]
                      + unary_potentials[edge[0]]
                      + pairwise.T)
            old_message = messages[e, 0].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(new_message)
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 0] = new_message
            update = new_message - old_message
            all_incoming[edge[1]] += update
            diff += np.abs(update).sum()

            # update message from edge[1] to edge[0]
            update = (all_incoming[edge[1]] - messages[e, 0]
                      + unary_potentials[edge[1]]
                      + pairwise)
            old_message = messages[e, 1].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(messages[e, 1])
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 1] = new_message
            update = new_message - old_message
            all_incoming[edge[0]] += update
            diff += np.abs(update).sum()
        if track_best:
            configuration = np.argmax(all_incoming + unary_potentials, axis=1)
            energy = compute_energy(
                unary_potentials=unary_potentials,  # yapf
                pairwise_potentials=pairwise_potentials,
                edges=edges,
                labels=configuration)
            if energy > best_energy:
                best_energy = energy
                best_configuration = configuration
        if diff < tol:
            break
    if track_best:
        return best_configuration
    else:
        return np.argmax(all_incoming + unary_potentials, axis=1)
