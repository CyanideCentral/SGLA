import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import logging
import warnings
import time

logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = np.identity(n,dtype='float32') - L
    try:
        evals, evecs = sparse.linalg.eigsh(X, min(rank, n//2), which=which)
    except sparse.linalg.ArpackError as e:
        print(e)
        evals, evecs = sparse.linalg.eigsh(X, rank, which=which, ncv=min(n, 1000))
    D_rt_inv = np.diag(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = np.diag(np.sqrt(evals)).dot(D_rt_invU.T).T
    Y = np.log(np.maximum((X @ X.T) * vol / b, 1.))
    return Y

def svd_deepwalk_matrix(X, dim):
    from sklearn.utils.extmath import randomized_svd
    u, s, v = randomized_svd(X,dim,n_oversamples=1,n_iter=1)
    return sparse.diags(np.sqrt(s)).dot(u.T).T

def netmf(A, dim, rank=32, window=10, negative=1.0):
    vol = float(A.sum())
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=rank, which="LA")
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    return deepwalk_embedding
