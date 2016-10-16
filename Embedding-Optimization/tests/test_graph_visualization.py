# import sys
# from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from rsc.utils import visualize_graph
from  sklearn.neighbors import kneighbors_graph
from  rsc.data.empirical_data import getMoon
if __name__ == '__main__':
    x, y = getMoon(n_sample=100, noise=0.1)
    print x.shape
    aff_mat = kneighbors_graph(x, n_neighbors=4, include_self=False)
    visualize_graph(affinity_matrix=aff_mat, dataX=x, dataY=y, neighbor_visualization=True)
    import matplotlib.pyplot as plt

    plt.show()