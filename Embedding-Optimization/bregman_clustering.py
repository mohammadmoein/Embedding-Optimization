import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def energy(H, A, A_c, lap_regularizer_cof=1, H_regularizer_cof=1,l1penalty=False):
    tmp = A - A_c
    lap = laplacian(A, normed=False)
    if l1penalty:
        return np.trace(tmp.dot(tmp.T)) + lap_regularizer_cof * np.trace(
        H.T.dot(lap).dot(H)) + H_regularizer_cof * np.trace(H.dot(H.T))
    return np.trace(tmp.dot(tmp.T)) + lap_regularizer_cof * np.trace(
        H.T.dot(lap).dot(H)) #+ H_regularizer_cof * np.trace(H.dot(H.T))


def solve_H(A, P, B, r, cluster=2, optimal=False,l1penalty=False,cofl1=0.1,Q=0,b=0):
    lap = laplacian(A, normed=False)
    if optimal == True:
        return np.linalg.eig(lap, eigvals=(0, cluster - 1))
    if l1penalty:
        return np.linalg.solve(2*lap+cofl1+r,r * (P - B)+cofl1*(Q-b))
    return np.linalg.solve(2* lap.T  + r * np.identity(lap.shape[0]), r * (P - B))


def solve_A(H, A_c,lap_regularizer_coff=1):
    diag_grad = (H ** 2).sum(axis=1).reshape(-1, 1)
    diag_grad = np.tile(diag_grad, (1, A_c.shape[1]))
    res =lap_regularizer_coff* H.dot(H.T) + 2 * A_c - lap_regularizer_coff*diag_grad
    return res / 2.0
def solve_Q(H,b,cofl1,mu):
    tmp = H+b
    threshold = 1.0/(cofl1*mu)
    indx = np.abs(tmp) <= threshold
    tmp[indx] = 0
    indx = np.abs(tmp) > threshold
    tmp[indx] = tmp[indx] - np.sign(tmp[indx]) * threshold
    return tmp


params = {'MAIN_ENERGY' :0,'H_VARIATION':1,'P_VARIATION':2}
colors = np.array([sns.xkcd_rgb["pale red"],sns.xkcd_rgb["denim blue"]])
def sp_bregman_l1penalty(A,data_x,data_y,iteration=100,r=0.001,n_cluster = 2,mu=0.1,l1cof=1.0):
    embds = []
    A_c = A.copy()
    # initialization
    H = np.arange(data_x.shape[0] * n_cluster).reshape(data_x.shape[0], n_cluster)
    P = H
    B = np.zeros(shape=H.shape)
    Q = H
    b = B
    values = np.zeros( (len(params),iteration), dtype=np.float)
    for iter in range(iteration):
        embds.append(H)
        H = normalize(H)
        current_energy = energy(H, A, A_c, lap_regularizer_cof=1,l1penalty=True,H_regularizer_cof=mu)
        values[params['MAIN_ENERGY'], iter] = current_energy
        H_new = solve_H(A, P, B, r=r, cluster=n_cluster, optimal=False,l1penalty=True,cofl1=l1cof,Q=Q,b=b)
        H_diff = np.linalg.norm(H_new - H)
        values[params['H_VARIATION'], iter] = H_diff
        H = H_new
        y_k = H + B
        u, d, v = np.linalg.svd(y_k, full_matrices=True)
        P_new = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
        print np.linalg.norm(P_new.dot(P_new.T) - P.dot(P.T))
        P = P_new
        Q = solve_Q(H,b,cofl1=l1cof,mu=mu)
        b = b+H-Q

        B = B + H - P
        # A = solve_A(H,A_c,lap_regularizer_coff=0.1)
        # A = 0.5*(A+A.T)
        # some
        P_diff = np.linalg.norm(np.identity(P.shape[0]) - P.dot(P.T))
        values[params['P_VARIATION'], iter] = P_diff

    kmeans = KMeans(n_cluster)
    pred = kmeans.fit_predict(normalize(H))
    emb = normalize(H)
    return values,emb,pred,embds

def sp_bregman(A,data_x,data_y,iteration=100,r=0.001,n_cluster = 2):
    embds = []
    A_c = A.copy()
    #initialization
    H = np.arange(data_x.shape[0] * n_cluster).reshape(data_x.shape[0], n_cluster)
    P = H
    B = np.zeros(shape=H.shape)


    values = np.zeros( (len(params),iteration), dtype=np.float)

    # H_variation = np.zeros(iteration,dtype=np.float)
    for iter in range(iteration):
        embds.append(H)
        # fname = '_tmp%03d.png' % iter
        # plt.scatter(H[:,0],H[:,1],c=colors[data_y])
        # plt.savefig(fname)
        # files.append(fname)
        current_energy = energy(H,A,A_c,lap_regularizer_cof=1)
        values[params['MAIN_ENERGY'],iter] = current_energy
        H_new = solve_H(A,P,B,r=r,cluster=n_cluster,optimal=False)
        H_diff = np.linalg.norm(H_new-H)
        values[params['H_VARIATION'],iter] = H_diff
        H = H_new
        y_k = H + B
        u, d, v = np.linalg.svd(y_k, full_matrices=True)
        P_new = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
        print np.linalg.norm(P_new.dot(P_new.T)-P.dot(P.T))
        P = P_new

        H = H_new
        B = B + H - P
        # A = solve_A(H,A_c,lap_regularizer_coff=0.1)
        # A = 0.5*(A+A.T)
        #some
        P_diff = np.linalg.norm(np.identity(P.shape[0]) - P.dot(P.T))
        values[params['P_VARIATION'],iter] = P_diff

    kmeans = KMeans(n_cluster)
    pred = kmeans.fit_predict(normalize(H))
    emb = normalize(H)
    return values,emb,pred,embds
def sp_sklearn(data_x,data_y,n_cluster,n_negihbors):
    from sklearn.cluster import SpectralClustering
    clf = SpectralClustering(n_clusters=n_cluster,affinity='nearest_neighbors',n_neighbors=n_negihbors).fit_predict(data_x)
    return clf
if __name__ == '__main__':
    NOISE = 0.2
    N_SAMPLE = 500
    data_x, data_y = make_moons(n_samples=N_SAMPLE,noise=NOISE)
    data_x_noisy, data_y_noisy = make_moons(n_samples=N_SAMPLE, noise=NOISE)
    ITERATION = 10
    NEIGHBOR = 5
    R = 0.1
    MU = 10
    L1COF = 1.0
    #adj matrix
    adj_matrix = kneighbors_graph(data_x, n_neighbors=NEIGHBOR, include_self=False).toarray()
    adj_matrix = 0.5 * (adj_matrix + adj_matrix.T)
    #bregman
    # values,emb,bg_pred,embd_spaces = sp_bregman_l1penalty(adj_matrix,data_x,data_y,ITERATION,r=R,n_cluster=2,mu=MU,l1cof=L1COF)
    values,emb,bg_pred,embd_spaces = sp_bregman(adj_matrix,data_x,data_y,ITERATION,r=R,n_cluster=2)
    #sklearn
    sk_pred = sp_sklearn(data_x,data_y,n_cluster=2,n_negihbors=NEIGHBOR)
    #plotting

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0,0].set_title('Dataset')
    ax[0,0].scatter(data_x[:,0],data_x[:,1],c =colors[data_y])
    ax[0,1].set_title('Spectral Clustering')
    ax[0,1].scatter(data_x[:,0],data_x[:,1],c = colors[sk_pred])
    ax[0,2].set_title('Bregman CLustering')
    ax[0,2].scatter(data_x[:,0],data_x[:,1],c = colors[bg_pred])
    fig, ax = plt.subplots(nrows=4, ncols=1)
    ax[0].set_title('Embedding Space')
    ax[0].scatter(emb[:,0],emb[:,1],c=colors[bg_pred])

    ax[1].set_title('Main Energy')
    ax[1].plot(range(ITERATION),values[params['MAIN_ENERGY']],label='Main Energy')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Energy')
    ax[2].set_title('H variation')
    ax[2].plot(range(ITERATION),values[params['H_VARIATION']],label='H variation')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('H norm diff')
    ax[3].set_title('P diff to identity')
    ax[3].plot(range(ITERATION),values[params['P_VARIATION']],label='P.dot(P.T)')
    ax[3].set_xlabel('Iteration')
    ax[3].set_ylabel('P norm diff')
    plt.legend()
    plt.show()

    # animation
    # from moviepy.video.io.bindings import mplfig_to_npimage
    # import moviepy.editor as mpy
    # # for idx,elements in enumerate(embd_spaces):
    # def make_frame_mpl(index):
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     ax.set_title('emb at %d' %index)
    #     elements = normalize(embd_spaces[int(index)])
    #     ax.scatter(elements[:,0],elements[:,1],c=colors[data_y])
    #     return  mplfig_to_npimage(fig)
    #
    # print len(embd_spaces)
    # animation = mpy.VideoClip(make_frame_mpl, duration=len(embd_spaces))
    # animation.write_videofile("spaces4_penalty_no_ag.mp4", fps=1)
    #
