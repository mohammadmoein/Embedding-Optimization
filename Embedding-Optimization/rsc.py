import timeit
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import seaborn as sns
from scipy.sparse.linalg import lobpcg
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import datetime

BREGMAN_KMEANS_WOLF = 'bregman_kmeans_wolf'

BREGMAN_KMEANS = 'bregman_kmeans'

EIGENVALUE_INITIALIZATION = 'optimal'

BREGMAN_ADJ = 'bregman_adj'
MAIN_ENERGY = "V1"
BREGMAN = 'bregman_spectral'
MY_SPECTRAL = 'normal_spectral'
SKLEARN_SPECTRAL = 'sklearn_spectral'
NEAREST_NEIGHBOR_TOKENS = 'near'
LAPLACIAN_UNNORMALIZED_TOKENS = 'unnormalized'
LAPLACIAN_NORMALIZED_TOKENS = 'normalized'
SPARSE_TYPE = 'csc'
viz_array = []

def frank_wolfe_kmeans(SAMPLE_SIZE,N_CLUSTER,MAX_ITERATION,x,Y=None,Z=None):
    X=x.T
    Y = np.zeros((SAMPLE_SIZE, N_CLUSTER)) if Y is None else Y
    Z = np.zeros((N_CLUSTER, SAMPLE_SIZE)) if Z is None else Z
    Y[0, :] = 1 if Y is None else Y[0,:]
    Z[0, :] = 1 if Z is None else Z[0,:]

    for iteration in range(MAX_ITERATION):
        # Y_UPDATE
        with np.errstate(divide='ignore'):
            L = (np.log(np.ma.array(Y, mask=(Y <= 0)))).filled(0)
        L += 1
        G_Y = 2 * (X.T.dot(X).dot(Y).dot(Z).dot(Z.T) - X.T.dot(X).dot(Z.T)) - L
        G_Z = 2 * (Y.T.dot(X.T).dot(X).dot(Y).dot(Z) - Y.T.dot(X.T).dot(X))
        for opt_iter in range(100):
            min_index_Y = np.argmin(G_Y, axis=0)
            min_index_Z = np.argmin(G_Z, axis=0)
            cof = 2.0 / (iteration + 2)
            update_matrix_Y = np.zeros_like(Y)
            update_matrix_Z = np.zeros_like(Z)

            update_matrix_Y[min_index_Y, np.arange(update_matrix_Y.shape[1])] = 1
            update_matrix_Z[min_index_Z, np.arange(update_matrix_Z.shape[1])] = 1
            Y = Y + cof * (update_matrix_Y - Y)
            Z = Z + cof * (update_matrix_Z - Z)
    return Y,Z
class RobustSpectralClustering:
    def __init__(self, n_cluster=2, n_init=10,
                 affinity='near', n_neighbor=5, dense=False):
        self.n_cluster = n_cluster
        self.n_init = n_init
        self.affinity = affinity
        self.n_neighbor = n_neighbor
        self.dense = dense

    def build_affinity_matrix(self, data_x):
        if self.affinity == NEAREST_NEIGHBOR_TOKENS:
            aff_mat = kneighbors_graph(data_x, n_neighbors=self.n_neighbor, mode='connectivity', include_self=False)
        val = 0.5 * (aff_mat + aff_mat.T)

        return val.toarray() if self.dense else val

    def laplacian(self, affinity_matrix, type=LAPLACIAN_UNNORMALIZED_TOKENS):
        row, col = affinity_matrix.shape

        degree_matrix = sp.spdiags(affinity_matrix.sum(axis=1).squeeze(), diags=0, m=row, n=col,
                                   format=SPARSE_TYPE) if sp.issparse(affinity_matrix) else np.diag(
            affinity_matrix.sum(axis=1).ravel(), k=0)

        if type == LAPLACIAN_UNNORMALIZED_TOKENS:
            return degree_matrix - affinity_matrix
        elif type == LAPLACIAN_NORMALIZED_TOKENS:
            degree_diagonal_sqrt = np.sqrt(degree_matrix.diagonal())
            degree_matrix_sqrt_inv = sp.spdiags(1.0 / (degree_diagonal_sqrt), diags=0, m=row, n=col,
                                                format=SPARSE_TYPE) if sp.issparse(degree_matrix) else np.diag(
                1.0 / (degree_diagonal_sqrt), k=0)

            return degree_matrix_sqrt_inv.dot(degree_matrix - affinity_matrix).dot(degree_matrix_sqrt_inv)

    def energy(self, laplacian, diffusion, version=MAIN_ENERGY):
        if version == MAIN_ENERGY:
            if sp.issparse(laplacian):
                diffusion_sparse = sp.csr_matrix(diffusion)
                val = (diffusion_sparse.T.dot(laplacian).dot(diffusion_sparse)).diagonal().sum()
            else:
                val = np.trace(diffusion.T.dot(laplacian).dot(laplacian))
            return val

    def solveH(self, laplacian, P, B, r):
        # bregman_distance = H - P + B
        # bregman_distance = np.trace(bregman_distance.T.dot(bregman_distance))
        # H_sparse = sp.csr_matrix(H) if sp.issparse(laplacian) else H
        # current_energy_respect_to_H = np.trace(H_sparse.T.dot(laplacian).dot(H_sparse)+0.5 * r * bregman_distance)
        # TODO MAybe sparse solver is not a good options
        sol = sp.linalg.spsolve(2 * laplacian + r * sp.identity(laplacian.shape[0]), r * (P - B))
        return sol
    def solveH_kmenas(self,laplacian,P,B,r,kmeans_cof,IC):
        sol = sp.linalg.spsolve(2 * laplacian + (r+kmeans_cof) * sp.identity(laplacian.shape[0]), r * (P - B)-kmeans_cof*IC)
        return sol

    def solve_A(self, H, A_c, lap_regularizer_coff=1):
        diag_grad = (H ** 2).sum(axis=1).reshape(-1, 1)
        diag_grad = np.tile(diag_grad, (1, A_c.shape[1]))
        res = lap_regularizer_coff * H.dot(H.T) + 2 * A_c - lap_regularizer_coff * diag_grad
        res /= 2.0
        res = 0.5 * (res.T + res)
        return sp.csc_matrix(np.round(res))
    def initial(self,laplacian=None,n_sample=None,constant_value=None,strategy=EIGENVALUE_INITIALIZATION):
        if strategy== EIGENVALUE_INITIALIZATION:
            if sp.issparse(laplacian):
                _, initial_diffusion_maps = sp.linalg.eigs(laplacian, k=self.n_cluster, which='SR')
                initial_diffusion_maps = initial_diffusion_maps.real
            else:
                _, initial_diffusion_maps = scipy.linalg.eigh(laplacian, eigvals=(0, self.n_cluster))
                initial_diffusion_maps = initial_diffusion_maps.real

        elif strategy=='random':
            initial_diffusion_maps = np.random.randn(n_sample,self.n_cluster)
        elif strategy=='cosntant':
            initial_diffusion_maps = np.empty((n_sample,self.n_cluster))
            np.fill(initial_diffusion_maps,constant_value)
        return initial_diffusion_maps
    def stoppin_criteria_check(self,variables_to_check,tol=1e-8):
        res =  np.array(map(lambda x:np.linalg.norm(x[1]-x[0])/np.linalg.norm(x[1]),variables_to_check))
        return (res<tol).any(),res
    def basis_vector(self,size,index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr
    def bregman_kmeans_wolfe(self,x,laplacian, r=300,kmeans_cof=100, iteration=100, tol=1e-5, verbose=False,initial_method=EIGENVALUE_INITIALIZATION):
        print 'bregman_kmenas_frank starts'
        H = self.initial(laplacian, n_sample=laplacian.shape[0], constant_value=None,
                         strategy=initial_method)  # np.random.randn(aff_mat.shape[0], self.n_cluster)
        # run kmeans to get initial
        
        kmeans = KMeans(self.n_cluster)
        kmeans.fit(H)
        centers = kmeans.cluster_centers_
        indicator = np.zeros((H.shape[0], self.n_cluster))
        indicator[np.arange(indicator.shape[0]), kmeans.predict(H)] = 1

        P = H
        B = np.zeros_like(H)
        energies = np.zeros(shape=iteration)
        list_of_embds = []
        for iter in range(iteration):
            list_of_embds.append(H)
            current_energy = self.energy(laplacian, diffusion=H)
            energies[iter] = current_energy
            H_new = self.solveH_kmenas(laplacian, P, B, r, kmeans_cof, indicator.dot(centers))
            stop_criteria = self.stoppin_criteria_check([(energies[iter - 1], current_energy), (H, H_new)])
            if stop_criteria <= tol:
                print 'stopping criteria satisfied at iteration %d' % (iter)
                break
            H = H_new
            #update kmeans

            #update Y with frank wolfe
            # Y_kmenas  = indicator.dot(np.linalg.inv(indicator.T.dot(indicator)))
           # Y_kmenas = np.zeros((H.shape[0],self.n_cluster))
            Z_kmeans = np.zeros((self.n_cluster,H.shape[0]))
            # Y_kmenas[0,:]=1
            # L_kmenas  = np.log(Y_kmenas+1)
            # kmeans_iteration=100
            # for kmeans_iter in range(kmeans_iteration):
            #     G = 2*(H.dot(H.T.dot(Y_kmenas.dot(indicator.T.dot(indicator))))-H.dot(H.T).dot(indicator))-L_kmenas
            #     j_prime =np.argmin(G,axis=0)
            #     E = np.zeros_like(Y_kmenas)
            #     E[j_prime,np.arange(E.shape[1])]=1
            #     Y_kmenas = Y_kmenas + (2.0/(kmeans_iter+2))*(E-Y_kmenas)
            # # kmeans_iter=0
            # for kmeans_iter in range(kmeans_iteration):
            indicator,weighted_indicator = frank_wolfe_kmeans(H.shape[0],self.n_cluster,MAX_ITERATION=100,x=x)
            # update P

            y_k = H + B
            u, d, v = np.linalg.svd(y_k)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            B = B + H - P

        if verbose:
            print "Is H equal to P (constraint satisfied)? %s. Diff:%f" % (np.allclose(P, H), np.linalg.norm(P - H))
            print "Is H equal orthogonal (difference of H^H norm to identity matrix)? %s" % np.linalg.norm(
                H.T.dot(H) - np.identity(H.shape[1]))

        return H, energies[:iter], list_of_embds

    def bregman_kmeans(self,laplacian, r=300,kmeans_cof=2, iteration=100, tol=1e-5, verbose=False,initial_method=EIGENVALUE_INITIALIZATION):
        print 'bregma_kmenas starts'
        H = self.initial(laplacian, n_sample=laplacian.shape[0], constant_value=None,
                         strategy=initial_method)  # np.random.randn(aff_mat.shape[0], self.n_cluster)
        #run kmeans to get initial
        kmeans = KMeans(self.n_cluster)
        kmeans.fit(H)
        centers = kmeans.cluster_centers_
        indicator  = np.zeros((H.shape[0],self.n_cluster))
        indicator[np.arange(indicator.shape[0]),kmeans.predict(H)]=1

        P = H
        B = np.zeros_like(H)
        energies = np.zeros(shape=iteration)
        list_of_embds=[]
        for iter in range(iteration):
            list_of_embds.append(H)
            current_energy = self.energy(laplacian, diffusion=H)
            energies[iter] = current_energy
            H_new = self.solveH_kmenas(laplacian, P, B, r,kmeans_cof,indicator.dot(centers))
            stop_criteria = self.stoppin_criteria_check([(energies[iter - 1], current_energy), (H, H_new)])
            if stop_criteria <= tol:
                print 'stopping criteria satisfied at iteration %d' % (iter)
                break
            H=H_new
            # update P

            y_k = H + B
            u, d, v = np.linalg.svd(y_k)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            B = B + H - P
            kmeans.fit(H)
            centers = kmeans.cluster_centers_
            indicator = np.zeros((H.shape[0], self.n_cluster))
            indicator[np.arange(indicator.shape[0]), kmeans.predict(H)] = 1

        if verbose:
            print "Is H equal to P (constraint satisfied)? %s. Diff:%f" % (np.allclose(P, H), np.linalg.norm(P - H))
            print "Is H equal orthogonal (difference of H^H norm to identity matrix)? %s" % np.linalg.norm(
                H.T.dot(H) - np.identity(H.shape[1]))

        return H, energies[:iter], list_of_embds

    def bregman_with_A(self, aff_mat, laplacian_type, r=1e-10, iteration=1000, tol=1e-5, movie=False, verbose=False,initial_method=EIGENVALUE_INITIALIZATION):
        Ac = aff_mat.copy()
        laplacian = self.laplacian(aff_mat, type=laplacian_type)
        H = self.initial(laplacian,n_sample=aff_mat.shape[0],constant_value=None,strategy=initial_method)#np.random.randn(aff_mat.shape[0], self.n_cluster)
        P = H
        B = np.zeros_like(H)
        energies = np.zeros(shape=iteration)
        list_of_affmat = []
        for iter in range(iteration):

            current_energy = self.energy(laplacian, diffusion=H)
            energies[iter] = current_energy

            # stop_criteria = np.abs(energies[iter] - energies[iter - 1]) / np.abs(energies[iter])
            H_new = self.solveH(laplacian, P, B, r)
            # Update Ac
            H=H_new
            list_of_affmat.append(aff_mat)
            aff_mat_new = self.solve_A(H, Ac)
            stop_criteria_eval,res = self.stoppin_criteria_check([(energies[iter - 1], current_energy), (H,H_new),(aff_mat,aff_mat_new)])
            if stop_criteria_eval <= tol:
                print 'stopping criteria satsfied at iteration %d with respect to %s'% (iter,res)
                break

            H=H_new
            aff_mat=aff_mat_new
            # update P

            y_k = H + B
            u, d, v = np.linalg.svd(y_k)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            B = B + H - P

            laplacian = self.laplacian(aff_mat, type=laplacian_type)

        if verbose:
            print "Is H equal to P (constraint satisfied)? %s %f" % (np.allclose(P, H), np.linalg.norm(P - H))
            print "Is H equal orthogonal (difference of H^H norm to identity matrix)? %s" % np.linalg.norm(
                H.T.dot(H) - np.identity(H.shape[1]))
        return H, energies[:iter], list_of_affmat
    def bregman(self, laplacian, r=300, iteration=100, tol=1e-5, verbose=False):
        print 'bregman starts'
        H = self.initial(laplacian=laplacian,n_sample=laplacian.shape[0],constant_value=None,strategy=EIGENVALUE_INITIALIZATION)
        P = H
        B = np.zeros_like(H)
        list_of_embds = []
        energies = np.zeros(shape=iteration)
        for iter in range(iteration):
            list_of_embds.append(H)
            current_energy = self.energy(laplacian, diffusion=H)
            energies[iter] = current_energy
            H_new = self.solveH(laplacian, P, B, r)
            stop_criteria,vals = self.stoppin_criteria_check([(energies[iter - 1], current_energy), (H,H_new)],tol=tol)
            H = H_new

            if stop_criteria:
                print 'stopping criteria satisfied at iteration %d' % (iter)
                break
            # update P

            y_k = H + B
            u, d, v = np.linalg.svd(y_k)
            P = u.dot(np.eye(N=u.shape[1], M=v.shape[0])).dot(v)
            B = B + H - P

        if verbose:
            print "Is H equal to P (constraint satisfied)? %s. Diff:%f" % (np.allclose(P, H), np.linalg.norm(P - H))
            print "Is H equal orthogonal (difference of H^H norm to identity matrix)? %s" % np.linalg.norm(
                H.T.dot(H) - np.identity(H.shape[1]))

        return H, energies[:iter+1], list_of_embds

    def train(self, data_x, method=MY_SPECTRAL, data_y=None, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, verbose=True,
              movie=True):
        aff_mat = self.build_affinity_matrix(data_x=data_x)
        lap = self.laplacian(affinity_matrix=aff_mat, type=laplacian_type)
        if method == MY_SPECTRAL:
            print 'My spectral starts'
            if sp.issparse(lap):
                lambads, diffusion_maps = sp.linalg.eigs(lap, k=self.n_cluster, which='SR')
            else:
                lambads, diffusion_maps = scipy.linalg.eigh(lap, eigvals=(0, self.n_cluster))
        elif method == SKLEARN_SPECTRAL:
            clf = SpectralClustering(n_clusters=self.n_cluster)
            pred = clf.fit_predict(data_x)
            return pred
        elif method == BREGMAN:
            diffusion_maps, energies, list_of_embds = self.bregman(laplacian=lap,
                                                                   tol=1e-5,r=300, verbose=True)
            viz_array.append({'y_data': energies, 'title': 'Energy_Bregman'})
            if movie: animation_maker(list_of_embds, data_y)
        elif method== BREGMAN_KMEANS:
            diffusion_maps, energies, list_of_embds = self.bregman_kmeans(laplacian=lap,
                                                                   tol=1e-10, r=200, kmeans_cof=2,
                                                                    verbose=True,iteration=100,initial_method=EIGENVALUE_INITIALIZATION)
            viz_array.append({'y_data': energies, 'title': 'Energy_Bregman_kmenas'})
        elif method == BREGMAN_KMEANS_WOLF:
            diffusion_maps, energies, list_of_embds = self.bregman_kmeans_wolfe(x=data_x, laplacian=lap,
                                                                          tol=1e-10, r=200, kmeans_cof=100,
                                                                          verbose=True, iteration=100,
                                                                          initial_method=EIGENVALUE_INITIALIZATION)
            viz_array.append({'y_data': energies, 'title': 'Energy_Bregman_kmenas'})
        elif method == BREGMAN_ADJ:
            diffusion_maps, energies, list_of_adj = self.bregman_with_A(aff_mat=aff_mat, laplacian_type=laplacian_type,
                                                                        tol=1e-10, verbose=True)
            diffusion_maps, energies,list_of_embds = self.bregman(laplacian=self.laplacian(list_of_adj[-1], type=laplacian_type),
                                                    laplacian_type=laplacian_type, tol=1e-10, verbose=True)
            viz_array.append({'y_data': energies, 'title': method})
            movie_maker(list_of_graph=list_of_adj, data_x=data_x, data_y=data_y)
        diffusion_maps = np.real(diffusion_maps)

        print method, 'Energy', self.energy(lap, diffusion_maps)

        diffusion_maps = normalize(diffusion_maps) if laplacian_type == LAPLACIAN_NORMALIZED_TOKENS else diffusion_maps
        kmeans = KMeans(self.n_cluster)
        pred = kmeans.fit_predict(diffusion_maps)
        return pred, diffusion_maps


def animation_maker(list_of_embds, data_y):
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy
    def make_frame(i):
        fig, ax = plt.subplots()
        fig.suptitle('frame %d with GT coloring' % (i))
        i = np.int(i)
        element = list_of_embds[i]
        n_colors = np.unique(data_y).shape[0]
        colors = np.array(sns.color_palette('hls', n_colors).as_hex())
        ax.scatter(element[:, 0], element[:, 1], c=colors[data_y], label=[str(i) for i in data_y])
        ax.set_xlim(
            [element[:, 0].min() - np.abs(element[:, 0].mean()), element[:, 0].max() + np.abs(element[:, 0].mean())])
        ax.set_ylim(
            [element[:, 1].min() - np.abs(element[:, 1].mean()), element[:, 1].max() + np.abs(element[:, 1].mean())])
        customs_color_map = mpl.colors.ListedColormap(colors)
        m = cm.ScalarMappable(cmap=customs_color_map)
        m.set_array(data_y)
        fig.colorbar(m, ticks=range(n_colors))
        ret_value = mplfig_to_npimage(fig)
        plt.close(fig)
        return ret_value

    animation = mpy.VideoClip(make_frame, duration=len(list_of_embds))
    animation.write_videofile("res/movie/embedding%s.mp4" % datetime.datetime.now(), fps=1, codec='mpeg4')


def movie_maker(list_of_graph, data_x, data_y):
    from utils import visualize_graph
    a = visualize_graph(list_of_graph[0], data_x, data_y, neighbor_visualization=False, edge_visualization=True).gcf()
    a.suptitle('first_graph')
    a = visualize_graph(list_of_graph[1], data_x, data_y, neighbor_visualization=False, edge_visualization=True).gcf()
    a.suptitle('Last Graph')
    # # animation
    # from moviepy.video.io.bindings import mplfig_to_npimage
    # import moviepy.editor as mpy
    # # for idx,elements in enumerate(embd_spaces):
    # def make_frame_mpl(index):
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     ax.set_title('graph at %d' %index)
    #     elements = list_of_graph[int(index)]
    #     visualize_graph(elements,data_x,data_y,neighbor_visualization=False,edge_visualization=True,ax=ax,fig=fig)
    #     # ax.scatter(elements[:,0],elements[:,1],c=colors[data_y])
    #     return  mplfig_to_npimage(ax)
    #
    # print len(list_of_graph)
    # animation = mpy.VideoClip(make_frame_mpl, duration=len(list_of_graph))
    # animation.write_videofile("graph.mp4", fps=1)
    # #


def visualization(data, n_cols=3):
    n_rows = len(data) / n_cols + 1
    plt.figure()
    for idx, element in enumerate(data):
        plt.subplot(n_rows, n_cols, idx + 1)
        title = element.get('title', 'None')
        label = element.get('label', 'None')
        x_data = element.get('x_data', 'None')
        x_text = element.get('x_text', 'None')
        y_data = element.get('y_data', 'None')
        y_text = element.get('y_text', 'None')

        if not (x_data is 'None') and len(x_data.shape) > 1:
            n_colors = np.unique(y_data).shape[0]
            colors = np.array(sns.color_palette('hls', n_colors).as_hex())
            plt.scatter(x_data[:, 0], x_data[:, 1], c=colors[y_data], label=[str(i) for i in y_data])
            customs_color_map = mpl.colors.ListedColormap(colors)
            m = cm.ScalarMappable(cmap=customs_color_map)
            m.set_array(y_data)
            plt.colorbar(m, ticks=range(n_colors))


        else:
            color = sns.xkcd_rgb['pale red']
            marker_color = sns.xkcd_rgb['denim blue']

            plt.plot(y_data / y_data.max(), markersize=7, c=color, marker='D', markerfacecolor=marker_color, lw=4,
                     label=label)
            # plt.gca().set_aspect('equal')
            plt.gca().set_ylim([-1, 2 * (y_data / y_data.max()).max()])
            # plt.gca().set_xlim([-10,y_data.shape[0]])
            import matplotlib.ticker as plticker

            loc = plticker.MultipleLocator(base=0.1)  # this locator puts ticks at regular intervals
            plt.gca().yaxis.set_major_locator(loc)
            # loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
            # plt.gca().xaxis.set_major_locator(loc)
            plt.gca().axhline(y=0, color='k')
            plt.gca().axvline(x=0, color='k')
            plt.legend()
        plt.title(title)
        plt.xlabel(x_text)
        plt.ylabel(y_text)


def with_adj_main():
    SAMPLE_SIZE = 100
    data_x, data_y = make_moons(n_samples=SAMPLE_SIZE, noise=0.1)
    clf = RobustSpectralClustering(dense=False, n_neighbor=10)
    import timeit
    start = timeit.default_timer()
    # y_pred, emb = clf.train(data_x, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, data_y=data_y,method=BREGMAN)
    # print 'Bregman Method %f' % (timeit.default_timer() - start)
    # start = timeit.default_timer()
    # y_pred_opt, emb_opt = clf.train(data_x, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, method=MY_SPECTRAL)
    y_pred_adj, emb_adj = clf.train(data_x, data_y=data_y, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS,
                                    method=BREGMAN_ADJ)

    # print 'Eigenvalue method %f' % (timeit.default_timer() - start)
    # print np.allclose(emb_opt, emb), np.linalg.norm(emb_opt - emb)
    plt1 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Ground_Truth', 'y_data': data_y, 'y_text': 'X2'}
    # plt2 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Prediction', 'y_data': y_pred, 'y_text': 'X2'}
    plt2_1 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Prediction_adj', 'y_data': y_pred_adj, 'y_text': 'X2'}
    # plt3 = {'x_data': emb, 'x_text': 'X1', 'title': 'Embedding with GR Coloring_Bregman Method', 'y_data': data_y,
    #         'y_text': 'X2'}
    # plt4 = {'x_data': emb_opt, 'x_text': 'X1', 'title': 'Embedding with GR Coloring_Eigenvalue', 'y_data': data_y,
    #         'y_text': 'X2', 'label': 'Eigenvalue decomposition'}
    plt5 = {'x_data': emb_adj, 'x_text': 'X1', 'title': 'Embedding with GR Coloring_Bregman_adj', 'y_data': data_y,
            'y_text': 'X2'}

    # viz_array.extend([plt1, plt2, plt3, plt4,plt2_1,plt5])
    viz_array.extend([plt1, plt2_1, plt5])
    visualization(viz_array)
    plt.show()

# def bregman_kmeans():
def without_adj_main():
    SAMPLE_SIZE = 100
    data_x, data_y = make_moons(n_samples=SAMPLE_SIZE,noise=0.1)
    clf = RobustSpectralClustering(dense=False)

    start = timeit.default_timer()
    y_pred_wolfe,emb_eolf = clf.train(data_x, data_y=data_y, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, method=BREGMAN_KMEANS_WOLF,
                  movie=False)
    y_pred, emb = clf.train(data_x, data_y=data_y, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, method=BREGMAN,
                            movie=False)
    print 'Bregman Method takes %f' % (timeit.default_timer() - start)
    start = timeit.default_timer()
    y_pred_opt, emb_opt = clf.train(data_x, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS, method=MY_SPECTRAL)
    y_pred_bregman_kmeans, emb_bregman_kmeans = clf.train(data_x, laplacian_type=LAPLACIAN_UNNORMALIZED_TOKENS,
                                                              method=BREGMAN_KMEANS_WOLF)
    print 'Eigenvalue method takes %f' % (timeit.default_timer() - start)
    print 'Are embedding spaces equal? %s. Diff is %f'%(np.allclose(emb_opt, emb), np.linalg.norm(emb_opt - emb))
    plt1 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Ground_Truth', 'y_data': data_y, 'y_text': 'X2'}
    plt2 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Prediction by Bregman', 'y_data': y_pred, 'y_text': 'X2'}
    plt2_2 = {'x_data': data_x, 'x_text': 'X1', 'title': 'Prediction by Bregman_kmeans', 'y_data': y_pred_bregman_kmeans, 'y_text': 'X2'}

    plt3 = {'x_data': emb, 'x_text': 'X1', 'title': 'Embedding with GT Coloring_Bregman Method', 'y_data': data_y,
            'y_text': 'X2'}

    plt4 = {'x_data': emb_opt, 'x_text': 'X1', 'title': 'Embedding with GT Coloring_Eigenvalue', 'y_data': data_y,
            'y_text': 'X2', 'label': 'Eigenvalue decomposition'}
    plt5 = {'x_data': emb_bregman_kmeans, 'x_text': 'X1', 'title': 'Embedding with GT Coloring_Bregman_kmeans', 'y_data': data_y,
            'y_text': 'X2', 'label': 'Eigenvalue decomposition'}
    viz_array.extend([plt1, plt2,plt2_2, plt3, plt4,plt5])
    visualization(viz_array)
    plt.show()


if __name__ == '__main__':
  
    without_adj_main()
    # with_adj_main()