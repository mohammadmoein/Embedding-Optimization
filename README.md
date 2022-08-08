# Robust Spectral Clustering 
Graph modeling appears as an effective and promising method to learn complex structures and
relationships hidden in data. Several spectral-based methods have been developed to cluster data into
different groups based on a graph model. Many of these methods assume that the given graph is not corrupted.
Therefore, the accuracy of these algorithms highly depends on the quality of the graph; that is, how well it
captures the similarity information between data points. On the contrary, they suffer severely in the
presence of noise or outliers in the data set.

The focus of this project is to design algorithms that are more robust to noise and outliers by performing
joint optimization on graphs and spectral embedding. This work attempts to develop a flexible framework
towards unifying clustering and adjacency learning. The work is designed to address two scenarios; learn
the graph from scratch or refine (denoise) a given graph such that it represents a similarity in a better
fashion. The solution to the methods of this thesis are achieved through the framework of convex
optimization. This framework offers the flexibility to use desired loss and regularization functions over
variables along with efficient algorithms to optimize. The graph learning process is conducted with the
help of integrating ideas proposed in the context of subspace learning, low-rank modeling and matrix
completion into spectral clustering.

Furthermore, we show how to minimize the spectral clustering objective (trace minimization) using split-
based methods such as Bregman and ADMM, instead of eigenvalue decomposition. This leads to a more
adjustable spectral clustering, allowing to apply different regularizers over the embedding. Our
experimental analysis confirms the huge potential of the proposed methods and benefits of their
flexibility.
![Embedding evolution landscape](Embedding-Optimization/res/spaces.gif)

The above figure shows how the embedding will start two decompose to two clusters after several iterations which lead to a space where vanilla kmeans does a great job at decomposing the datapoints. 
## Modeling & Optimization
Let, $A \in \reals^{n\times n}$ be a given adjacency matrix, where $n$ is the number of data points. The main idea of RSC is that the adjacency (similarity) includes noise. Noise is translated to incorrectly connected edges in RSC. Thus, RSC attempts to find bad edges and remove them in the hope of having better similarity matrix. 
RSC decompose $A$ in two factors:
$$
\begin{gather*}
 A = A^g + A^c\\
 A^g = A^{g^T},A^c = A^{c^T}, 
\end{gather*}
$$
where $A^c \in \reals^{n \times n}$ represents the corruption (noise) occurred in observed $A$ and $A^g \in \reals^{n \times n}$ represents true adjacency matrix. It is desired to do the clustering on $A^g$, thus the problem reduced on how to find $A^g$. 


An informative graph essentially determines the potentials of the vast graph-oriented learning algorithms. This section is about approaches in which they construct graph by taking advantage of the overall contextual information instead of only pairwise Euclidean distance and local information. Compared with the conventional $k$-nn graph and $\epsilon$-ball graph, these methods have following advantages:

 - not relying on the local neighborhood 
 - robustness to noise and outliers
 - forcing certain structure and adaptive sparsity

### RPCA Spectral Clustering

Given a matrix $A$ generated as the sum of two components
$$
\begin{equation}
A = L + E, 
\end{equation}
$$
where $L$ is an ideal low-rank matrix and $E$ represents the noise. Then, RPCA constructs the following optimization problem,
$$
\begin{gather}
 \argmin\limits_{L,E} rank(L) + \lambda \|E\|_0 \\
 A = L +E
\end{gather}
$$
This constraint usually substituted by nuclear norm $|L|_*$, the summation of singular values. So, the problem  becomes:
$$
\begin{gather}
 \argmin\limits_{L,E} \|L\|_* + \lambda \|E\|_0 \\
 A = L +E
\end{gather}
$$
,which can be solved efficiently with ADMM method. 

RPCA Spectral clustering (RPCASC) is formulated as follows. Given adjacency matrix $A_c \in \reals^{n \times n}$,then
$$
\begin{gather}
 \argmin \limits_{H,Z,E} \|Z\|_* + \lambda\|E\|_1 + \mu \quad Tr (H^T(diag(Z\textbf{1})-Z)H)  \\
 A_c = Z + E \\
 H^TH = I.
\end{gather}
$$
$$
\begin{gather}
 \argmin \limits_{H,Z,E,J} \|J\|_* + \lambda\|E\|_1 + \mu\quad Tr (H^T(diag(Z\textbf{1})-Z)H)  \\
 A_c = Z + E \\
 H^TH = I\\
 Z = J.
\end{gather}
$$
#### Augmented Lagrangian Function

$$
\begin{align}
 &L(H,Z,E,J,Y_1,Y_2) = \|J\|_* + \lambda\|E\|_1 + \mu \quad Tr (H^T(diag(Z\textbf{1})-Z)H) + \\&<Y_1, A_c - Z - E> + <Y_2, Z-J> 
 +\frac{\beta}{2}\left[\|Z-J\|_F^2 + \|A_c - Z - E\|_F^2\right].\nonumber
\end{align}
$$
- **Updating J** 
$$
\begin{gather}
 \|J\|_* + \frac{\beta}{2} \|J - (Z + Y_2/\beta)|^2_F.
\end{gather}
$$
Which is equal to ```prox``` of nuclear norm called _singular value thresholding_.
$$
prox_{\lambda\|.\|_*} (A) = \sum_{1}^{n} (\sigma_i - \lambda)_{+} u_i v_i^T
$$,
where $A = \sum_{1}^{n} \sigma_i  u_iv_i^T$ is the singular value decomposition of $A$.

- **Updating E** 
$$
\begin{gather}
 \lambda \|E\|_1 + \frac{\beta}{2}\|E - (A_c - Z + Y_1/\beta)\|^2_F,
\end{gather}
$$

- **Updating Z** 
$$
\begin{align}
 &\quad Tr(H^T\mathbb{diag}(Z\textbf{1})H) - \quad Tr(H^TZH) +\\& <Y_1, A_c - Z - E> + <Y_2, Z-J> 
 +\frac{\beta}{2}\left[\|Z-J\|_F^2 + \|A_c - Z - E\|_F^2\right] \nonumber,
\end{align}
$$

- **Updating H**
$$
\begin{gather}
 \min\quad Tr(H^TL_zH) \\ \nonumber
 H^TH = I,
\end{gather}
$$
where $L_z$ denotes a Laplacian built upon adjacency matrix induced by $0.5*(|Z|+|Z^T|)$. 
- **Updating Lagrange**
$$
\begin{gather}
\nonumber
 Y_1 = Y_1 + \beta(A_c - J -E)\\
  \nonumber
 Y_2 = Y_2 + \beta(Z - J ).
\end{gather}
$$

This concludes the RPCASC algorithm that jointly optimizes over the embedding $H$ and adjacency matrix $Z$. 