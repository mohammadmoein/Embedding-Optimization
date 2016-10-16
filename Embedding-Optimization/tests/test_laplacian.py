import sys
from os.path import dirname
from os import getcwd
sys.path.append(dirname(dirname(getcwd())))
import numpy as np
from scipy.sparse.csgraph import laplacian
from rsc.rsc import RobustSpectralClustering,LAPLACIAN_NORMALIZED_TOKENS,LAPLACIAN_UNNORMALIZED_TOKENS
from sklearn.datasets import make_moons

x,y = make_moons(100)
a = RobustSpectralClustering()
aff = a.build_affinity_matrix(x)
aff = 0.5*(aff+aff.T)
# aff=aff.toarray()
my_lap_unormalized = a.laplacian(aff,LAPLACIAN_UNNORMALIZED_TOKENS)
my_lap_normalized = a.laplacian(aff,LAPLACIAN_NORMALIZED_TOKENS)
scip_normalized = laplacian(aff,normed=True)
scip_unormalized = laplacian(aff,normed=False)
print np.allclose(my_lap_normalized.toarray(),scip_normalized.toarray())
print np.allclose(my_lap_unormalized.toarray(),scip_unormalized.toarray())
# print np.allclose(my_lap_normalized,scip_normalized)
# print np.allclose(my_lap_unormalized,scip_unormalized)
