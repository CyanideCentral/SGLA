import resource
import config as config
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from spectral import discretize
from evaluate import clustering_metrics
from datasets import load_data
from scipy.optimize import minimize
import argparse
import warnings
from evaluate import ovr_evaluate
from sparse_dot_mkl import dot_product_mkl
warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)

def parse_args():
    p = argparse.ArgumentParser(description='Set parameter')
    p.add_argument('--dataset', type=str, default='dblp', help='dataset name (e.g.: acm, dblp, imdb)')
    p.add_argument('--scale', action='store_true', help='configurations for large-scale data (mageng/magphy)')
    p.add_argument('--embedding', action='store_true', help='run embedding task')
    p.add_argument('--verbose', action='store_true', help='print verbose logs')
    p.add_argument('--knn', type=int, default=10, help='k neighbors except imdb=500, yelp=200' )
    p.add_argument('--embed_dim', type=int, default=64, help='embedding output demension')
    p.add_argument('--tmax', type=int, default=100, help='maximum number of iterations for COBYLA optimizer')
    p.add_argument('--epsilon', type=float, default=0.001, help='convergence threshold for COBYLA optimizer')
    p.add_argument('--gamma', type=float, default=0.5, help='coefficient of weight regularization')
    
    args = p.parse_args()
    config.verbose = args.verbose
    config.embedding = args.embedding
    config.knn = args.knn
    config.embed_dim = args.embed_dim
    config.tmax = args.tmax
    config.epsilon = args.epsilon
    config.gamma = args.gamma
    return args

def SGLA(dataset):
    num_clusters = dataset['k']
    n = dataset['n']
    nv = dataset['nv']
    g_adjs = [g.astype('float32') for g in dataset['graphs']]
    features = [f.astype('float32') for f in dataset['features']]
    view_weights = np.full(nv, 1.0/nv)
    knn_adjs = []
    start_time = time.time()

    for X in features:
        import faiss
        if sp.issparse(X):
            X=X.astype(np.float32).tocoo()
            ftd = np.zeros(X.shape, X.dtype)
            ftd[X.row, X.col] = X.data
        else :
            ftd = X.astype(np.float32)
            ftd = np.ascontiguousarray(ftd)
        faiss.normalize_L2(ftd)
        if config.scale:
            index = faiss.index_factory(ftd.shape[1], "IVF1000,PQ40", faiss.METRIC_INNER_PRODUCT)
            index.train(ftd)
        else:
            index = faiss.index_factory(ftd.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(ftd)
        distances, neighbors = index.search(ftd, config.knn+1)
        knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
        knn.setdiag(0.0)
        knn = knn + knn.T
        knn_adjs.append(knn + knn.T)

    if config.verbose:
        print(f'KNN graph construction time: {time.time()-start_time:.2f}s')
    g_dvs = [sp.diags(np.asarray(g_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(g_adjs))]
    knn_dvs = [sp.diags(np.asarray(knn_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(knn_adjs))]
    for dv in [*g_dvs, *knn_dvs]:
        dv.data[dv.data==0] = 1
        dv.data = dv.data**-0.5
    computed_g = [g_dvs[i]@dot_product_mkl(g_adjs[i], g_dvs[i],cast = True) for i in range(len(g_adjs))]
    computed_knn = [knn_dvs[i]@dot_product_mkl(knn_adjs[i], knn_dvs[i],cast = True) for i in range(len(knn_adjs))]
    
    # Linear operator of multi-view Laplacian
    def mv_lap(mat):
        if config.scale:
            product = np.zeros_like(mat, dtype='float32')
        else:
            product = np.zeros(mat.shape, dtype='float32')
        iv = 0
        for i in range(len(g_adjs)):
            product += view_weights[iv] * dot_product_mkl(computed_g[i],mat,cast=True)
            iv += 1
        for i in range(len(knn_adjs)):
            product += view_weights[iv] * dot_product_mkl(computed_knn[i],mat,cast=True)
            iv += 1
        return mat-product
    lapLO = sla.LinearOperator((n, n), matvec=mv_lap, rmatvec=mv_lap, dtype='float32')

    # Direct optimize
    opt_time = time.time()
    def eig_obj(w):
        view_weights[:-1] = w
        view_weights[-1] = 1.0 - np.sum(w)
        eig_val = sla.eigsh(lapLO, num_clusters+1, which='SM', tol=config.eig_tol, maxiter=1000, return_eigenvectors=False)
        eig_val = eig_val.real
        eig_val.sort()
        return eig_val[num_clusters-1] / eig_val[num_clusters] - eig_val[1] + config.gamma*np.power(np.asarray(view_weights),2).sum()
    
    w_constraint = [{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)}, {'type': 'ineq', 'fun': lambda w: min(w)}, {'type': 'ineq', 'fun': lambda w: 1.0-max(w)}]
    opt_w = minimize(eig_obj, np.full((nv-1), 1.0/nv), method='COBYLA', tol=config.epsilon, constraints=w_constraint, options={'maxiter': config.tmax, 'rhobeg': config.cobyla_rhobeg, 'disp': config.verbose})
    if config.verbose:
        print(f"opt_time: {time.time()-opt_time}")

    if config.embedding:
        delta=sp.eye(dataset['n'],dtype="float32")-mv_lap(sp.eye(dataset['n'],format="csr",dtype="float32"))
        if config.scale:
            from sketchne import sketchne_graph
            emb = sketchne_graph(delta, dim = config.embed_dim, spec_propagation=False, window_size=10, eta1=32, eta2=32, eig_rank=64, power_iteration=20)
        else:
            from netmf import netmf
            emb = netmf(delta, dim = config.embed_dim)
        embed_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        emb_results = ovr_evaluate(emb, dataset['labels'])
        print(f"Time: {embed_time:.3f}s RAM: {int(peak_memory_MBs)}MB")
    else: # clustering
        eig_val, eig_vec = sla.eigsh(lapLO, num_clusters+1, which='SM', tol=config.eig_tol, maxiter=1000)
        predict_clusters, _ = discretize(eig_vec[:, :num_clusters])
        cluster_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        cm = clustering_metrics(dataset['labels'], predict_clusters)
        acc, nmi, f1, _, ari, _ = cm.evaluationClusterModelFromLabel()
        print(f"Acc: {acc:.3f} F1: {f1:.3f} NMI: {nmi:.3f} ARI: {ari:.3f} Time: {cluster_time:.3f}s RAM: {int(peak_memory_MBs)}MB")

    if config.verbose:
        print(f"Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(args.dataset)
    if args.dataset.startswith("mag"):
        config.scale = True
    SGLA(dataset)


