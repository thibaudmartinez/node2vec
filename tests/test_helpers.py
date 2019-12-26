import numpy as np
from gensim.models import Word2Vec

from node2vec.helpers import wv_to_numpy_array


class TestWvToNumpyArray:
    def test_success(self):
        walks = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
            [5, 6, 7, 8, 9, 10],
            [6, 7, 8, 9, 10, 0],
            [7, 8, 9, 10, 0, 1],
            [8, 9, 10, 0, 1, 2],
            [9, 10, 0, 1, 2, 3],
            [10, 0, 1, 2, 3, 4],
        ]
        walks = [list(map(str, x)) for x in walks]

        word2vec_model = Word2Vec(
            sentences=walks,
            size=8,
            window=2,
            min_count=0,
            sg=1,  # use skip-gram
            workers=1,
            iter=1,
        )

        embeddings = wv_to_numpy_array(word2vec_model.wv)

        for key in word2vec_model.wv.vocab.keys():
            node_id = int(key)
            wv_as_np = np.array(word2vec_model.wv[key], dtype=np.float32)

            np.testing.assert_array_equal(embeddings[node_id], wv_as_np)
