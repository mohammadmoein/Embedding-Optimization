#Frank-Wolfe implementaiton of k-means
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import seaborn as sns
sns.set()
from sklearn.datasets import make_blobs
SAMPLE_SIZE = 100
MAX_ITERATION = 100
INNER_OPTIMIZATION = 10
N_CLUSTER = 3

x,y = make_moons(n_samples=SAMPLE_SIZE)
x,y = make_blobs(n_samples=SAMPLE_SIZE,centers=3)
X = x.T

#Initialization
Y = np.zeros((SAMPLE_SIZE,N_CLUSTER))
Z = np.zeros((N_CLUSTER,SAMPLE_SIZE))
Y[0,:] = 1
Z[0,:] = 1

for iteration in range(MAX_ITERATION):
    #Y_UPDATE
    with np.errstate(divide='ignore'):
        L = (np.log(np.ma.array(Y, mask=(Y <= 0)))).filled(0)
    L += 1 
    G_Y = 2*(X.T.dot(X).dot(Y).dot(Z).dot(Z.T) - X.T.dot(X).dot(Z.T)) - L
    G_Z = 2*(Y.T.dot(X.T).dot(X).dot(Y).dot(Z) - Y.T.dot(X.T).dot(X))
    for opt_iter in range(100):
        min_index_Y = np.argmin(G_Y, axis=0)
        min_index_Z = np.argmin(G_Z, axis=0)
        cof = 2.0/(iteration+2)
        update_matrix_Y = np.zeros_like(Y)
        update_matrix_Z = np.zeros_like(Z)

        update_matrix_Y[min_index_Y,np.arange(update_matrix_Y.shape[1])] = 1
        update_matrix_Z[min_index_Z, np.arange(update_matrix_Z.shape[1])] = 1
        Y = Y + cof * (update_matrix_Y-Y)
        Z = Z + cof * (update_matrix_Z-Z)
fig, ax = plt.subplots(2,1)
ax[0].scatter(x[:,0],x[:,1], c=y)

Z[Z<0.5]=0
Z[Z>=0.5]=1

ax[1].scatter(x[:,0],x[:,1], c=np.argmax(Z,axis=0)*1.0)
plt.show()

    
