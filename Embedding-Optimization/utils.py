import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sm
import seaborn as sns

# CONSTANT
LINEWIDTH = 1


def visualize_graph(affinity_matrix, dataX, dataY, neighbor_visualization=False, edge_visualization=False):
    node_size = int(dataX.shape[0]*20/1000)
    number, dims = dataX.shape
    # n_graph = len(affinity_matrix)

    if neighbor_visualization is True: edge_visualization = True
    number_unique_label = np.unique(dataY).shape[0]
    node_colors = sns.husl_palette(number_unique_label)
    customs_color_map = mpl.colors.ListedColormap(node_colors)
    node_colors = np.array(node_colors)

    # Color Definition
    hidden_node_color = [sns.xkcd_rgb["light grey"]]
    edge_colors = sns.xkcd_rgb["medium green"]

    nonzero_indices = affinity_matrix.nonzero() if sm.issparse(affinity_matrix) else np.where(affinity_matrix != 0)
    plt.figure()

    if dims == 2:
        plt.scatter(dataX[:, 0], dataX[:, 1], c=node_colors[dataY - 1], cmap=customs_color_map, s=node_size, zorder=10,
                    picker=neighbor_visualization,
                    label=[str(i) for i in dataY])
        if edge_visualization:
            plt.plot([dataX[nonzero_indices[0], 0], dataX[nonzero_indices[1], 0]],
                     [dataX[nonzero_indices[0], 1], dataX[nonzero_indices[1], 1]], c=edge_colors, lw=LINEWIDTH)
        m = cm.ScalarMappable(cmap=customs_color_map)
        m.set_array(dataY)
        plt.colorbar(m, ticks=range(number_unique_label))
        # TODO add 3d graph support
        # elif dims ==3:
    if neighbor_visualization:

        plt.figure("Neighbor_Graph")

        def onpick(event):
            ind = event.ind
            if len(ind) > 1:
                return
            plt.figure("Neighbor_Graph").clear()
            row = affinity_matrix[ind].toarray() if sm.issparse(affinity_matrix) else affinity_matrix[ind]

            maskedarray = row.astype(np.bool).squeeze()
            maskedarray[ind] = True
            # -1 due to labelin is 1,2 but the indexing is zero and 1

            plt.plot([np.tile(dataX[ind, 0], maskedarray.sum()), dataX[maskedarray, 0]],
                     [np.tile(dataX[ind, 1], maskedarray.sum()), dataX[maskedarray, 1]], c=edge_colors,
                     lw=5 * LINEWIDTH)
            plt.scatter(dataX[maskedarray, 0], dataX[maskedarray, 1], c=node_colors[dataY[maskedarray] - 1],
                        cmap=customs_color_map,
                        s=node_size,
                        picker=neighbor_visualization, alpha=0.7)
            maskedarray = ~maskedarray
            plt.scatter(dataX[maskedarray, 0], dataX[maskedarray, 1], c=hidden_node_color, cmap=customs_color_map,
                        s=node_size,
                        picker=neighbor_visualization, alpha=0.7)
            plt.figure("Neighbor_Graph").canvas.draw()
            print 'onpick3 scatter:', ind, np.take(dataX[:, 0], ind), np.take(dataX[:, 1], ind)

        plt.figure("data2").canvas.mpl_connect('pick_event', onpick)
    return plt

def adj_matrix_visualization(adj_matrix):
    plt.figure("Matrix Visualization")
    if sm.issparse(adj_matrix):
        plt.spy(adj_matrix)
        # plt.colorbar()
    else:
        plt.matshow(adj_matrix, cmap=plt.get_cmap("cool"), interpolation='none', vmin=0, vmax=1)
        plt.colorbar()
    plt.grid()


if __name__ == '__main__':
    from data import getMoon

    data = getMoon(n_sample=100, noise=0.1)
    x, y = data
    from  sklearn.neighbors import kneighbors_graph

    aff_mat = kneighbors_graph(x, n_neighbors=15, include_self=False)
    visualize_graph(aff_mat,dataX=x,dataY=y,neighbor_visualization=False,edge_visualization=True)
    # adj_matrix_visualization(aff_mat)
    plt.show()
