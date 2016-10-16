import numpy as np
import scipy.sparse as sm
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph


class AffintyMatrix:
    # X is m observation by n feature

    def __init__(self, X):
        self.x = X
        self.affmat = None

    def _euclideanDistance(self):
        return squareform(pdist(self.x, metric="euclidean", p=2))

    def _seuclideanDistance(self):
        return squareform(pdist(self.x, metric="seuclidean", p=2))

    def epsGraph(self, eps=1, metric='squaredeuclidean'):
        distance = self._seuclideanDistance() if metric == 'squaredeuclidean' else self._euclideanDistance()
        indices = distance < eps
        affmat = np.zeros_like(distance)
        affmat[indices] = 1
        return sm.csr_matrix(affmat)

    def k_nearestneighborGraph(self, k=10):
        return kneighbors_graph(self.x, n_neighbors=k, include_self=False,mode='connectivity')

    def fullyconnectedGraph(self, sigma=1):
        euc = self._euclideanDistance() / (2.0 * sigma * 2)
        a = np.exp(-euc)
        np.fill_diagonal(a, 0)
        return a

    def get(self, k=10, eps=1, type='k_nearest', sigma=1):
        if type == 'full':
            adj_matrix = self.fullyconnectedGraph(sigma=sigma)

        elif type == 'k_nearest':
            adj_matrix = self.k_nearestneighborGraph(k=k)

        else:
            adj_matrix = self.epsGraph(eps=eps)
        #we assumed neighbor information is symmetric
        return 0.5*(adj_matrix+adj_matrix.T)
