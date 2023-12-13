import math
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator
import os
import sys

class GraRep(Estimator):
    r"""An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`_
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.

    Args:
        dimensions (int): Number of individual embedding dimensions. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 5.
        seed (int): SVD random seed. Default is 42.
    """

    def __init__(
        self, dimensions: int = 32, iteration: int = 10, order: int = 5, seed: int = 42
    ):
        self.dimensions = dimensions
        self.iterations = iteration
        self.order = order
        self.seed = seed

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array(
            [1.0 / graph.degree[node] for node in range(graph.number_of_nodes())]
        )
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.csr_matrix((values, (index, index)), shape=shape)
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating a tuple with the normalized adjacency matrix.

        Return types:
            * **(A_hat, A_hat)** *(Tuple of SciPy arrays)* - Normalized adjacency matrices.
        """
        A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()), dtype=np.bool_).tocsr()
        D_inverse = sparse.csr_matrix(self._create_D_inverse(graph), dtype=np.float32)
        A_hat = D_inverse.dot(A)
        self._save_matrix(A_hat, "A_hat")
        self._save_matrix(A_hat, "A^0")
    
    def threshold(self, matrix, threshold, minmax):
        # Apply threshold of minimum value to be kept in matrix

        # if matrix is coo matrix
        if isinstance(matrix, sparse.coo_matrix):
            if minmax == "min":
                mask = matrix.data < threshold
            elif minmax == "max":
                mask = matrix.data > threshold
            matrix.data = matrix.data[mask]
            matrix.row = matrix.row[mask]
            matrix.col = matrix.col[mask]
        # if matrix is csr matrix
        elif isinstance(matrix, sparse.csr_matrix):
            if minmax == "floor":
                mask = matrix.data < threshold
            elif minmax == "ceiling":
                mask = matrix.data > threshold
            matrix.data[mask] = 0
            matrix.eliminate_zeros()
        else:
            raise ValueError("Matrix type not supported")
        return matrix
    
    def _save_matrix(self, matrix, name):
        # Save matrix to file
        if isinstance(matrix, np.ndarray):
            np.save(f"Data/{self._name}/tmp/{name}.npy", matrix)
        elif isinstance(matrix, sparse.csr_matrix):
            sparse.save_npz(f"Data/{self._name}/tmp/{name}.npz", matrix)
            
    def _load_matrix(self, name):
        if name[0] == 'W':
            return np.load(f"Data/{self._name}/tmp/{name}.npy", allow_pickle=True)
        else:
            matrix = sparse.load_npz(f"Data/{self._name}/tmp/{name}.npz")
            return matrix
    
    def _create_target_matrix(self, k):
        """
        Creating a log transformed target matrix.

        Return types:
            * **target_matrix** *(SciPy array)* - The PMI matrix.
        """
        _A_tilde = self._load_matrix(f"A^{k-1}")
        _A_hat = self._load_matrix("A_hat")
        # print data type of each matrix
        # as the order increases, the adjacency matrix raised to that power becomes denser,
        # which means more non-zero elements. This increases the memory requirements,
        # as sparse matrix representations are most efficient when the matrix is mostly zeros.
        _A_tilde = sparse.csr_matrix(_A_tilde.dot(_A_hat), dtype=np.float32)
        del _A_hat
        
        _A_tilde = self.threshold(_A_tilde, threshold=1e-4, minmax="floor")
        print(f"Size of A^{k} in kB: {_A_tilde.data.nbytes/1024}")
        self._save_matrix(_A_tilde, f"A^{k}")
            
        # _A_tile acts as the target matrix from now on
        _A_tilde.data = np.log(_A_tilde.data) - math.log(_A_tilde.shape[0])
        _A_tilde = self.threshold(_A_tilde, threshold=0, minmax="ceiling")

        return _A_tilde

    def _create_single_embedding(self, target_matrix, k):
        """
        Fitting a single SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(
            n_components=self.dimensions, n_iter=self.iterations, random_state=self.seed
        )
        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        self._save_matrix(embedding, f"W^{k}")
        
    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a GraRep model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._name = graph.name
        graph = self._check_graph(graph)
        if not os.path.exists(f"Data/{self._name}/tmp"):
            os.makedirs(f"Data/{self._name}/tmp")

        self._create_base_matrix(graph)
        # delete the variable global variable graph
        del graph
        
        for k in range(1, self.order+1):
            if os.path.exists(f"Data/{self._name}/tmp/W^{k}.npy"):
                continue
            target_matrix = self._create_target_matrix(k)
            self._create_single_embedding(target_matrix, k)


    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        # delete all variables except embedding        
        embeddings = []
        for i in range(1, self.order+1):
            embedding = self._load_matrix(f"W^{i}")
            embeddings.append(embedding)
        embeddings = np.concatenate(embeddings, axis=1)
        return embeddings

if __name__ == "__main__":
    # test with random graph
    graph = nx.fast_gnp_random_graph(5000, 0.05)
    graph.name = 'test'
    grarep = GraRep(order=5, dimensions=2)
    grarep.fit(graph)
    embedding = grarep.get_embedding()
