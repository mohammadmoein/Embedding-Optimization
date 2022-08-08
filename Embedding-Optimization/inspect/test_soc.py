import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.sparse.csgraph import laplacian
import cPickle as pkl
import scipy
import networkx as nx
from sklearn.datasets import make_moons
from rsc import RobustSpectralClustering

ITERATION = 100
R = 300
N_CLUSTER =2
DESNSE = False

def get_Matrix(mode="moon",size=100):
    if mode == 'moon':
        x, y = make_moons(size,noise=0.1)
        from sklearn.neighbors import graph
        adj_matr = graph.kneighbors_graph(x, n_neighbors=5, include_self=False)
        adj_matr = 0.5 * (adj_matr + adj_matr.T)
        return adj_matr
    elif mode == 'random':
        return np.random.randn(size,size)
    elif mode == 'zeros':
        return np.zeros((size,size))
    else:
        return np.identity(n=size)


def main_energy(H,L):
    if sp.issparse(L) :
        return np.trace(H.T.dot(L.dot(H)))
    else:
        return np.trace(H.T.dot(L).dot(H))
def solve_H(L,P,B,R):
    if sp.issparse(L):
        return linalg.spsolve(2*L+R*sp.identity(L.shape[0]),R*(P-B))
    else:
        return np.linalg.solve(2*L+R*np.identity(L.shape[0]),R*(P-B))
def lap(matrix,type = 'normalized'):
    return  laplacian(matrix,normed=type=="normalized")
def get_initial(n_sample,n_cluster,matrix=None,strategy='random'):
    if strategy=='random':
        return np.random.randn(n_sample,n_cluster)
    elif strategy =='zeros':
        return np.zeros((n_sample,n_cluster))
    elif strategy == 'optimal':
        _,res = scipy.linalg.eigh(matrix, eigvals=(0, n_cluster - 1)) if DESNSE else  sp.linalg.eigs(matrix, k=n_cluster, which='SR')
        return res.real
if __name__ == '__main__':
    mat = get_Matrix(mode="moon")
    L = lap(mat,type="unnormalized")
    if DESNSE: L= L.toarray()
    H = get_initial(L.shape[0],n_cluster=N_CLUSTER,matrix=L,strategy='random')
    P = H
    B  = np.zeros_like(H)
    energies = np.zeros(ITERATION)
    for i in range(ITERATION):
        currentenergy = main_energy(H,L)
        energies[i] = currentenergy# if currentenergy!=0 else np.finfo.eps
        # stop_criteria = np.abs(energies[i] - energies[i - 1]) / (np.abs(energies[i]))

        # stop_criteria = np.linalg.norm(H-solve_H(L,P,B,R))
        # print stop_criteria
        # if stop_criteria < 1e-5:
        #     print 'Stopping criteria satisfied at iteration %d'%(i)
        #     break
        # H = a.solveH(L,P,B,R)
        H = solve_H(L,P,B,R)
        y_k = H + B
        u, d, v = np.linalg.svd(y_k)
        P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
        B = B + H - P
    # rsc_H,_,_ = a.bregman(L,r=R,iteration = ITERATION)
    print "R is %f and ITERATION is %d"%(R,ITERATION)
    print "Is H equal to P?%s"%(np.allclose(H,P)),'diif',np.linalg.norm(H-P)
    print "diff of H^TH by identity %f"%(np.linalg.norm(H.T.dot(H)-np.identity(H.shape[1])))
    lambdas,diff_map= scipy.linalg.eigh(L, eigvals=(0, N_CLUSTER - 1)) if DESNSE else  sp.linalg.eigs(L, k=N_CLUSTER, which='SR')
    # tmp = H.T.dot(H)
    # tmp =np.round(tmp)
    # print H.T.dot(H)
    # print ">>>>>"
    # print tmp
    # print "mesure",np.abs(tmp).sum().sum()-np.diag(np.abs(tmp)).sum()
    print "opt diff by H-bregman is %f"%(np.linalg.norm(H - diff_map))
    print "opt energy %f and bregman energy %f"%(main_energy(diff_map.real,L),main_energy(H,L))
    # from spectral.algorithms import orthogonalize
    # u=orthogonalize(H)
    # # print 'polar decomposition',u.dot(u.T)
    # plt.figure('polar')
    # plt.scatter(u[:,0],u[:,1],c=y,s=40)

    # print 'Embedding-Optimization energy',main_energy(rsc_H,L)
    # plt.scatter(rsc_H[:,0],rsc_H[:,1])
    plt.figure('bregman')

    # plt.scatter(H[:,0],H[:,1],c=y,s=40)
    plt.scatter(H[:,0],H[:,1],s=40)
    plt.figure('optimal')
    plt.scatter(diff_map.real[:,0],diff_map.real[:,1],s=40)
    # plt.plot(range(ITERATION),energies/np.abs(energies).max())
    plt.show()