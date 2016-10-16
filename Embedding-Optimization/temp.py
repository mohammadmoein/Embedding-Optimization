# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
a=np.random.randn(30,2)
kmeans = KMeans(3)

kmeans.fit(a)
print kmeans.cluster_centers_
centers=kmeans.cluster_centers_
print centers.shape
q=np.zeros((a.shape[0],3))
indicator = kmeans.predict(a)
q[np.arange(q.shape[0]),indicator]=1
kmeans.fit(q)
print kmeans.cluster_centers_
print indicator
print q
# import matplotlib.pyplot as plt
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# r = 1
# pi = np.pi
# cos = np.cos
# sin = np.sin
# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
# x = r*sin(phi)*cos(theta)
# y = r*sin(phi)*sin(theta)
# z = r*cos(phi)
# ax.plot_wireframe(
#     x, y, z,  cmap=cm.jet)
#
# H = np.mgrid[-1:1.1:0.2, -1:1.1:0.2].reshape(2, -1).T
# z = H[:,0]**2+H[:,1]**2
# z = H[:,0]**2+H[:,0]*H[:,1]*2+H[:,1]**2
#
#
#
# ax.plot_trisurf(H[:,0], H[:,1], z, cmap=cm.jet, linewidth=0.2,alpha=0.6)
#
# plt.show()