import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np
import os
import pickle as pkl
import pandas as pd
import networkx as nx

def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z

def get_proximity_matrix(A, dist_A, k_hop=1, sigma=25):
    """
    Get a proximity matrix from adjacency matrix and distance matrix

    :param A: 2d dimension of array
    :return: 2d numpy array
    """
    assert A.ndim == 2
    # construct a graph from existing Adjacency Matrix
    G = nx.from_numpy_matrix(A)
    # Get k hops of the adjacency matrix
    nodes = list(range(A.shape[0]))
    # apply exponential operation
    proximity_matrix = np.exp(-1 * dist_A ** 2 / sigma)

    W = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for node_i in nodes:
        length = nx.single_source_shortest_path_length(G, node_i, k_hop)
        for node_j, length_j in length.items():
            if length_j > 0:
                W[node_i, node_j] = proximity_matrix[node_i, node_j]
        W[node_i, node_i] = proximity_matrix[node_i, node_i]
    W = np.maximum(W, W.transpose())

    return W

def distance_scipy_spatial(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def distance_lshforest(z, k=4, metric='cosine'):
    """Return an approximation of the k-nearest cosine distances."""
    assert metric is 'cosine'
    lshf = sklearn.neighbors.LSHForest()
    lshf.fit(z)
    dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
    assert dist.min() < 1e-10
    dist[dist < 0] = 0
    return dist, idx

# TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN
def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph. W should be symmetric and non-negative
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = scipy.sparse.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0) # Remove single-dimensional entries from the shape of an array.
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype)) # get a small number in case to be divided by zero
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]


def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U


def plot_spectrum(L, algo='eig'):
    """Plot the spectrum of a list of multi-scale Laplacians L."""
    # Algo is eig to be sure to get all eigenvalues.
    plt.figure()
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)
    plt.savefig('multi_spectrum.eps')


def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype)
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt  # Q[0, ...]


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt


def grid_graph(m, k=8, metric='euclidean', corners=False):
    z = grid(m)
    # sort the distance of the grid
    dist, idx = distance_sklearn_metrics(z, k=k, metric=metric)
    A = adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, k*m**2//2))
    return A

def convert_edge_graph(directed_dict, random_node=True):
    # get a directed graph and convert it into undirected, why?
    # 1. In paper "Convolutional Neural Networks on Graphs with Fast
    # Localized Spectral Filtering", it deals with undirected and connected graph G.
    # 2. We just consider the spatial relation of these links, direction has not been utilized yet.

    di_graph = nx.from_dict_of_lists(directed_dict)
    undi_graph = di_graph.to_undirected()

    # get connected subgraphs
    graphs = list(nx.connected_component_subgraphs(undi_graph))
    nb_nodes = [len(graph_i.nodes()) for graph_i in graphs]
    max_nb = max(nb_nodes)
    max_index = nb_nodes.index(max_nb)
    max_sub_graph = graphs[max_index]

    # get the adjacency matrix of the largest sub graph
    nodes = max_sub_graph.nodes()
    adj = nx.adjacency_matrix(max_sub_graph)
    adj.setdiag(0)
    adj = adj.astype(np.float64)
    if random_node:
        adj = adj.todense()
        rnd_order = np.random.permutation(len(nodes))
        adj = adj[rnd_order, :]
        adj = adj[:, rnd_order]
        nodes = list(np.array(nodes)[rnd_order])
        # convert into a sparse format
        adj = scipy.sparse.csr_matrix(adj)
    adj.eliminate_zeros()
    nodes = [str(node) for node in nodes]

    return adj, nodes

def specify_node(link_id, row, col_name, added, node_id, dict_edge_node):
    nodes = []
    nodes.append(link_id)
    if row[col_name] is not None:
        in_tops = row[col_name].split(',')
        for in_top in in_tops:
            node = in_top + added
            nodes.append(node)
    cur_node_id = node_id
    node_exist = False
    for node in nodes:
        if node in dict_edge_node.keys():
            node_exist = True
            cur_node_id = dict_edge_node[node]
            break
    for node in nodes:
        dict_edge_node[node] = cur_node_id
    if node_exist:
        cur_node_id = node_id
    else:
        cur_node_id += 1

    return dict_edge_node, cur_node_id

def construct_node_graph(r_dir, random_node=True, engine=None):
    """
    convert a graph from edge connections to node connection

    :param r_dir: string, directory to store or obtain the results
    :param random_node: bool, whether to random shuffle the data
    :param engine: sqlalchemy sql server engine
    :return:
        dict_edge_node, a dictionary contains the mapping of edge node to real node
        link_ids, list of link ids used in this road network
        W, Adjacency matrix of the resulted graph
        L, Laplacian matrix of the resulted graph
        D, Degree matrix of the resulted graph
    """

    path_directed_dict = os.path.join(r_dir, 'kdd_dict_edge_node.pickle')
    path_links = os.path.join(r_dir, 'kdd_links.pickle')
    if not os.path.exists(path_directed_dict) or not os.path.exists(path_links) or True:
        conn = engine.connect()
        sql_links = "select link_id, in_top, out_top from links"
        df_links = pd.read_sql(sql_links, conn)
        conn.close()
        link_ids = df_links['link_id'].tolist()
        dict_edge_node = {}
        node_id = 0
        if random_node:
            df_links = df_links.iloc[np.random.permutation(len(df_links))]
        for index, row in df_links.iterrows():
            link_id = row['link_id']
            # Deal with the left side
            col_name = 'in_top'
            added = '_o'
            link_id_left = '{}_i'.format(link_id)
            dict_edge_node, node_id = specify_node(link_id_left,
                row, col_name, added, node_id, dict_edge_node)

            # Deal with the right side
            col_name = 'out_top'
            added = '_i'
            link_id_right = '{}_o'.format(link_id)
            dict_edge_node, node_id = specify_node(link_id_right,
                row, col_name, added, node_id, dict_edge_node)

        with open(path_directed_dict, 'wb') as f_di_dict:
            pkl.dump(dict_edge_node, f_di_dict)

        with open(path_links, 'wb') as f_links:
            pkl.dump(link_ids, f_links)
    else:
        with open(path_directed_dict, 'rb')  as f_di_dict:
            dict_edge_node = pkl.load(f_di_dict)
        with open(path_links, 'rb') as f_links:
            link_ids = pkl.load(f_links)

    # get a directed graph and convert it into undirected
    di_graph = nx.DiGraph()
    for link_id in link_ids:
        left_node = '{}_i'.format(link_id)
        right_node = '{}_o'.format(link_id)
        left_node_id = dict_edge_node[left_node]
        right_node_id = dict_edge_node[right_node]
        di_graph.add_edge(left_node_id, right_node_id)
    plt.show()
    nx.draw(di_graph, with_labels=True)
    plt.savefig("kdd_rn.jpg")
    nodes = di_graph.nodes()
    W = nx.adjacency_matrix(di_graph, nodes)
    W_row_sum = [sum(W.sum(axis=1).tolist(), [])]
    D = scipy.sparse.diags(W_row_sum, [0])
    L = D - W

    return dict_edge_node, link_ids, W, L, D

def construct_latex_matrix(A):
    for i in range(A.shape[0]):
        str1 = ''
        for j in range(A.shape[1]):
            str1 += str(A[i][j])
            if j < A.shape[1] - 1:
                str1 += '&\t'
        str1 += '\\\\'
        print(str1)
