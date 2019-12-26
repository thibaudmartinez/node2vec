import numpy as np


class EdgeListError(ValueError):
    pass


def check_edge_list(src_nodes, dst_nodes, edge_weights):
    """Checks that the input edge list is valid."""

    if len(src_nodes) != len(dst_nodes):
        raise EdgeListError("src_nodes and dst_nodes must be of same length.")

    if edge_weights is None:
        return

    if len(edge_weights) != len(src_nodes):
        raise EdgeListError("src_nodes and edge_weights must be of same length.")


class AdjacencyMatrixError(ValueError):
    pass


def check_adj_matrix(adj_matrix):
    """Checks that the input adjacency matrix is valid."""
    if adj_matrix.ndim != 2:
        raise AdjacencyMatrixError("The numpy array must be of dimension 2.")

    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise AdjacencyMatrixError("The matrix must be squared.")


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)


def wv_to_numpy_array(wv):
    vocab_keys = [int(key) for key in wv.vocab.keys()]
    embeddings = [wv[str(key)] for key in sorted(vocab_keys)]
    return np.array(embeddings, dtype=np.float32)
