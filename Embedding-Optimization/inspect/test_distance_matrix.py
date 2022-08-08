# import sys
# from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from rsc.utils import visualize_graph
from rsc.data.empirical_data import getMoon
from rsc.distance_matrix import AffintyMatrix

NUMBER_NEGHBOR = 3
if __name__ == '__main__':
    x,y = getMoon()
    aff_mat = AffintyMatrix(X=x)
    aff_mat = aff_mat.k_nearestneighborGraph(k=NUMBER_NEGHBOR)
    visualize_graph(affinity_matrix=aff_mat,dataX=x,dataY=y,neighbor_visualization=True)
    import matplotlib.pyplot as plt
    plt.show()

