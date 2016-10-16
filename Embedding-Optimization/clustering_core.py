import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from distance_matrix import AffintyMatrix
from utils import adj_matrix_visualization
from sklearn.preprocessing import normalize

class myspectralclustering():
    def __init__(self, n_cluster,type="k_nearest", sigma=1, n_neighbor=5):
        self.n_cluster = n_cluster
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        

    def laplacian(self, adj_matrix, type="unnormalized"):
        return sp.csgraph.laplacian(adj_matrix)
        row,col = adj_matrix.shape
        degree_matrix = sp.spdiags(adj_matrix.sum(axis=1).squeeze(),diags=[0],m=row,n=col)
        if type == "unnormalized":
             return degree_matrix-adj_matrix
    def bregman_fit(self,X,adj_type='k_nearest',iter=1000,r=1,threshold=0.000005):
        # adj_matrix = AffintyMatrix(X).get(self.n_neighbor,type=adj_type).toarray()
        from sklearn.neighbors import kneighbors_graph
        adj_matrix  =kneighbors_graph(X,include_self=False,n_neighbors=self.n_neighbor)
        # laplacianMatrix = csgraph.laplacian(adj_matrix,normed=False).toarray()
        adj_matrix = 0.5*(adj_matrix+adj_matrix.T)
        # laplacianMatrix =self.laplacian(adj_matrix,type="unnormalized").tocsc()
        laplacianMatrix =sp.csgraph.laplacian(adj_matrix)


        #Initializaiton

        # H=np.random.randn(X.shape[0],self.n_cluster)
        H=np.arange(X.shape[0]*self.n_cluster).reshape(X.shape[0],self.n_cluster)
        P = H
        row,_ = H.shape
        B = np.zeros(shape=H.shape)
        A_c = adj_matrix.copy()
        for i in range(iter):
            H_prev = H
            #TODO find away to do it in sparse

            # H = sp.linalg.spsolve(laplacianMatrix+laplacianMatrix.T+r*sp.identity(n=row),sp.csc_matrix(r*(P-B))).toarray()
            H = np.linalg.solve((laplacianMatrix+laplacianMatrix.T)+r*np.identity(n=row),r*(P-B))

            # diag_grad = (H ** 2).sum(axis=1).reshape(-1, 1)
            # diag_grad = np.tile(diag_grad, (1, A_c.shape[1]))
            # res = H.dot(H.T) + 2 * A_c - diag_grad
            # adj_matrix  = res/4.0
            # adj_matrix = 0.5*(adj_matrix+adj_matrix.T)
            # laplacianMatrix = sp.csgraph.laplacian(adj_matrix,normed=False)
            y_k = H + B
            u, d, v = np.linalg.svd(y_k, full_matrices=True)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            err = np.linalg.norm(H-H_prev)
            B = B + H - P
            if err<threshold:
                print 'threshold meets %f'%err
                break
        kmeans = KMeans(self.n_cluster)
        pred = kmeans.fit_predict(H)
        return pred,normalize(H)
    def energy(self,adjmatrix,H,copy,lap):
        return np.trace(H.T.dot(lap).dot(H))+np.trace((adjmatrix-copy).T.dot(adjmatrix-copy))

    def bregman_fit_adj(self,X,adj_type='k_nearest',iter=1000,r=1,threshold=0.000005):
        plt.figure('ener')
        adj_matrix = AffintyMatrix(X).get(self.n_neighbor, type=adj_type)
        adj_matrix = 0.5*(adj_matrix+adj_matrix.T)
        laplacianMatrix = self.laplacian(adj_matrix, type="unnormalized").toarray()

        # Initializaiton
        Ac = adj_matrix.copy()
        H = np.arange(X.shape[0] * self.n_cluster).reshape(X.shape[0], self.n_cluster)
        P = H
        row, col= H.shape
        B = np.zeros(shape=H.shape)
        #Adj parameter
        q = 1
        mu = 1
        cof = 1.0/(mu+q)

        for i in range(iter):
            lr = 0.1
            H_prev = H
            # TODO find away to do it in
            # H = sp.linalg.spsolve(laplacianMatrix + laplacianMatrix.T + r * sp.eye(laplacianMatrix.shape[0],laplacianMatrix.shape[1]),
            #                       sp.csc_matrix(r * (P - B))).toarray()
            H = np.linalg.solve((laplacianMatrix+laplacianMatrix.T)+r*np.identity(row),r*(P-B))
            #A
            grad_part = np.tile((H.dot(H.T)).sum(axis=0).reshape(-1,1),(1,adj_matrix.shape[1]))-H.dot(H.T)
            for i  in range(50):
                grad = grad_part+2*adj_matrix-2*Ac
                adj_matrix = adj_matrix-lr*grad
                lr/=4

            # adj_matrix = sp.csr_matrix(cof*(H.dot(H.T)-(H.dot(np.ones((H.shape[1],1))))+q*Ac.toarray()))
            adj_matrix = 0.5*(adj_matrix+adj_matrix.T)
            # adj_matrix[adj_matrix>=0.5]=1
            # adj_matrix[adj_matrix<0.5]=0

            laplacianMatrix = self.laplacian(adj_matrix, type="unnormalized")
            print self.energy(adj_matrix,H,Ac,laplacianMatrix)
            plt.scatter(i,self.energy(adj_matrix,H,Ac,laplacianMatrix))
            y_k = H + B
            u, d, v = np.linalg.svd(y_k, full_matrices=True)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            # if not np.allclose(np.dot(P.T, P), np.identity(n=P.T.shape[0])):
            #     print "Constrained subproblem failed"
            #     break
            # err = np.linalg.norm(H.T.dot(H).astype(int)-np.identity(H.shape[1]))
            err = np.linalg.norm(H - H_prev)
            B = B + H - P
            if err < threshold:
                # print 'diff',np.linalg.norm(H.T.dot(H) - np.identity(H.shape[1]))
                # eigen_value, emb = eigh(laplacianMatrix, eigvals=(0, self.n_cluster - 1))
                # print np.linalg.norm(emb-H)
                print 'threshold meets %f' % err
                break
        # adj_matrix_visualization(adj_matrix.toarray())
        # plt.show()
        # exit()
        kmeans = KMeans(self.n_cluster)
        pred = kmeans.fit_predict(H)
        return pred, normalize(H)

    def fit(self, X, affinity='k_nearest'):

        aff_matrix = AffintyMatrix(X)

        if affinity == 'full':
            adj_matrix = aff_matrix.fullyconnectedGraph(sigma=self.sigma)
        elif affinity == 'k_nearest':
            adj_matrix = aff_matrix.k_nearestneighborGraph(k=self.n_neighbor)
        else:
            adj_matrix = aff_matrix.epsGraph()

        adj_matrix=0.5*(adj_matrix+adj_matrix.T)
        if not np.allclose(adj_matrix.toarray(),adj_matrix.toarray().T):
            print ("bingo")
            exit()
        laplacianMatrix = np.diag(adj_matrix.dot(np.ones(adj_matrix.shape[0])))-adj_matrix
        # laplacianMatrix = csgraph.laplacian(adj_matrix,normed=False).tocsr()
        if sp.issparse(laplacianMatrix):
            eignen_value, emb = eigsh(laplacianMatrix, self.n_cluster, laplacianMatrix)
        else:
            eigen_value, emb = eigh(laplacianMatrix, eigvals=(0, self.n_cluster - 1))

        kmeans = KMeans(self.n_cluster)
        pred = kmeans.fit_predict(emb)
        return pred, normalize(emb)


if __name__ == '__main__':
    from data.empirical_data import getMoon

    np.random.seed(0)
    N_SAMPLE =100
    N_NOISE = 0.0001
    N_NOISE = 0.0001
    neighbor =5
    x, y = getMoon(n_sample=N_SAMPLE, noise=N_NOISE)#empricalData(n_sample=N_SAMPLE, noise=N_NOISE).getMoon()
    clustering = myspectralclustering(n_cluster=2,n_neighbor=neighbor)
    # y_pred,emb = clustering.bregman_fit_adj(X=x)
    # y_pred,emb = clustering.bregman_fit(X=x)
    y_pred,emb = clustering.bregman_fit(X=x)
    from utils import visualize_graph
    visualize_graph(AffintyMatrix(x).k_nearestneighborGraph(k=neighbor),x,y,neighbor_visualization=True)
    # adj_matrix_visualization(adj)
    # visualize_graph(aff,x,y,neighbor_visualization=True,edge_visualization=True)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Prediction')
    plt.scatter(x[:,0],x[:,1],c=y_pred)
    plt.subplot(2,2,2)
    plt.title('embedding_predicted labeling')
    plt.scatter(emb[:,0],emb[:,1],c=y_pred)
    plt.subplot(2,2,4)

    plt.title('embedding_true labeling')
    plt.scatter(emb[:, 0], emb[:, 1], c=y)
    plt.show()

