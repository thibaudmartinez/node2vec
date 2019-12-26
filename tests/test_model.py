import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from node2vec.helpers import AdjacencyMatrixError
from node2vec.helpers import EdgeListError
from node2vec.model import Node2Vec


class TestConstructor:
    @pytest.mark.parametrize(
        "dtype_node, dtype_weights",
        [
            (int, float),
            (np.int32, np.float32),
            (np.int64, np.float64),
            (np.int32, np.double)
        ]
    )
    def test_success_multiple_data_types(self, dtype_node, dtype_weights):
        src_nodes = np.array([0, 1, 0, 2, 2, 3], dtype=dtype_node)
        dest_nodes = np.array([1, 2, 2, 1, 3, 0], dtype=dtype_node)
        edge_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype_weights)

        Node2Vec(src_nodes, dest_nodes, edge_weights)

    def test_success_unweighted_graph(self):
        src_nodes = np.array([0, 1, 0, 2, 2, 3], dtype=np.int32)
        dest_nodes = np.array([1, 2, 2, 1, 3, 0], dtype=np.int32)

        Node2Vec(src_nodes, dest_nodes)

    def test_success_directed_graph(self):
        src_nodes = np.array([0, 1, 0, 2, 2, 3], dtype=np.int32)
        dest_nodes = np.array([1, 2, 2, 1, 3, 0], dtype=np.int32)
        edge_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)

        Node2Vec(src_nodes, dest_nodes, edge_weights, graph_is_directed=True)

    def test_failure_invalid_source_nodes_(self):
        src_nodes = np.array([0, 1], dtype=np.int32)
        dest_nodes = np.array([1, 2, 2, 1, 3, 0], dtype=np.int32)

        with pytest.raises(
            EdgeListError,
            match=r"src_nodes and dst_nodes must be of same length."
        ):
            Node2Vec(src_nodes, dest_nodes)

    def test_failure_invalid_weights(self):
        src_nodes = np.array([0, 1, 0, 2, 2, 3], dtype=np.int32)
        dest_nodes = np.array([1, 2, 2, 1, 3, 0], dtype=np.int32)
        edge_weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        with pytest.raises(
            EdgeListError,
            match=r"src_nodes and edge_weights must be of same length."
        ):
            Node2Vec(src_nodes, dest_nodes, edge_weights)


class TestFromNxGraph:
    def test_success_undirected_graph(self):
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        edges = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (9, 10)]

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        model = Node2Vec.from_nx_graph(graph)

        assert not model._graph_is_directed
        np.testing.assert_array_equal(
            model._edge_weights,
            np.ones(len(edges), dtype=np.double)
        )

    def test_success_directed_graph(self):
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        edges = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (9, 10)]

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        model = Node2Vec.from_nx_graph(graph)

        assert model._graph_is_directed
        np.testing.assert_array_equal(
            model._edge_weights,
            np.ones(len(edges), dtype=np.double)
        )

    def test_success_weighted_undirected_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([
            (0, 1, {"weight": 1}), (1, 2, {"weight": 2}), (2, 3, {"weight": 10})
        ])
        model = Node2Vec.from_nx_graph(graph)

        assert not model._graph_is_directed
        np.testing.assert_array_equal(
            model._edge_weights,
            np.array([1, 2, 10], dtype=np.double)
        )

    def test_success_weighted_directed_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([
            (0, 1, {"weight": 1}), (1, 2, {"weight": 2}), (2, 3, {"weight": 10})
        ])
        model = Node2Vec.from_nx_graph(graph)

        assert model._graph_is_directed
        np.testing.assert_array_equal(
            model._edge_weights,
            np.array([1, 2, 10], dtype=np.double)
        )


class TestFromAdjMatrix:
    @pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
    def test_success_multiple_data_types(self, dtype):
        adj_matrix = np.array([
            [0, 1, 2, 0],
            [0, 0, 1, 2],
            [1, 0, 0, 1],
            [1, 0, 0, 0]
        ], dtype=dtype)

        Node2Vec.from_adj_matrix(adj_matrix)

    def test_success_directed_graph(self):
        adj_matrix = np.array([
            [0, 1, 2, 0],
            [0, 0, 1, 2],
            [1, 0, 0, 1],
            [1, 0, 0, 0]
        ], dtype=np.int32)

        model = Node2Vec.from_adj_matrix(adj_matrix)
        assert model._graph_is_directed

    def test_undirected_graph(self):
        adj_matrix = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ], dtype=np.int32)

        model = Node2Vec.from_adj_matrix(adj_matrix)
        assert not model._graph_is_directed

    def test_failure_matrix_dim_3(self):
        adj_matrix = np.array([
            [[0, 1, 2, 1], [1, 0, 1, 2]],
            [[2, 1, 0, 1], [1, 2, 1, 0]]
        ], dtype=np.int32)

        with pytest.raises(
                AdjacencyMatrixError,
                match=r"The numpy array must be of dimension 2."
        ):
            Node2Vec.from_adj_matrix(adj_matrix)

    def test_failure_non_squared_matrix(self):
        adj_matrix = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [1, 2, 1, 0]
        ])

        with pytest.raises(
                AdjacencyMatrixError,
                match=r"The matrix must be squared."
        ):
            Node2Vec.from_adj_matrix(adj_matrix)


class TestSimulateWalks:
    def test_success_directed_graph(self):
        src_nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        dest_nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32)
        edge_weights = np.ones(len(src_nodes), dtype=float)

        adj_matrix = sparse.coo_matrix(
            (edge_weights, (src_nodes, dest_nodes))
        ).toarray()

        model = Node2Vec(src_nodes, dest_nodes, edge_weights, graph_is_directed=True)
        model.simulate_walks(walk_length=10, n_walks=10, rand_seed=19, workers=1)

        print(adj_matrix)

        for walk in model.walks:
            print(walk)
            for i in range(len(walk) - 1):
                src = walk[i]
                dest = walk[i + 1]
                assert adj_matrix[src, dest] == 1.0

    def test_success_undirected_graph(self):
        adj_matrix = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ], dtype=np.int32)

        model = Node2Vec.from_adj_matrix(adj_matrix)
        model.simulate_walks(walk_length=10, n_walks=10, rand_seed=19, workers=1)

        for walk in model.walks:
            for i in range(len(walk) - 1):
                src = walk[i]
                dest = walk[i + 1]
                assert adj_matrix[src, dest] != 0.0


class TestLearnEmbeddings:
    def test_success(self):
        src_nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        dest_nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int32)

        model = Node2Vec(src_nodes, dest_nodes, graph_is_directed=False)
        model._walks = np.array([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
            [5, 6, 7, 8, 9, 0],
        ])

        model.learn_embeddings(dimensions=10, context_size=5, workers=1, rand_seed=9)
