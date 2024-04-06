import resource
import config as config
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from spectral import discretize
from evaluate import clustering_metrics
from datasets import load_data
from scipy.optimize import fmin_cobyla
import argparse
from evaluate import ovr_evaluate
from sparse_dot_mkl import dot_product_mkl
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def parse_args():
    p = argparse.ArgumentParser(description='Set parameter')
    p.add_argument('--dataset', type=str, default='dblp', help='dataset name (e.g.: acm, dblp, imdb)')
    p.add_argument('--embedding', action='store_true', help='run embedding task')
    p.add_argument('--verbose', action='store_true', help='print verbose logs')
    p.add_argument('--knn_k', type=int, default=10, help='k neighbors except imdb=500, yelp=200' )
    p.add_argument('--embed_rank', type=int, default=32, help='eigsh rank in netmf' )
    args = p.parse_args()
    config.dataset = args.dataset
    config.verbose = args.verbose
    config.embedding = args.embedding
    config.knn_k = args.knn_k
    config.embed_rank = args.embed_rank
    return args

def SMGF_PI(dataset):
    num_clusters = dataset['k']
    n = dataset['n']
    nv = dataset['nv']
    graph_adjs = dataset['graphs']
    features = dataset['features']
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
        distances, neighbors = index.search(ftd, config.knn_k+1)
        knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
        knn.setdiag(0.0)
        knn = knn + knn.T
        knn_adjs.append(knn + knn.T)
    if config.verbose:
        print(f'KNN graph construction time: {time.time()-start_time:.2f}s')

    graph_dvs = [sp.diags(np.asarray(graph_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(graph_adjs))]
    knn_dvs = [sp.diags(np.asarray(knn_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(knn_adjs))]
    for dv in [*graph_dvs, *knn_dvs]:
        dv.data[dv.data==0] = 1
        dv.data = dv.data**-0.5
    
    sample_obj=[]
    sample_w = []
    centroid = np.full(nv, 1.0/nv)
    sample_w.append(centroid)
    for i in range(nv):
        vertex_i = np.zeros_like(centroid)
        vertex_i[i] = 1.0
        sample_w.append(0.5*(vertex_i + centroid))

    # Linear operator of multi-view Laplacian
    def mv_lap(mat):
        if config.scale:
            product = np.zeros_like(mat)
        else:
            product = np.zeros(mat.shape)
        iv = 0
        for i in range(len(graph_adjs)):
            product += view_weights[iv] * graph_dvs[i]@dot_product_mkl(graph_adjs[i], (graph_dvs[i]@mat), cast=True)
            iv += 1
        for i in range(len(knn_adjs)):
            product += view_weights[iv] * knn_dvs[i]@dot_product_mkl(knn_adjs[i], (knn_dvs[i]@mat), cast=True)
            iv += 1
        return mat-product
    lapLO = sla.LinearOperator((n, n), matvec=mv_lap)
    for num in range(len(sample_w)):
        view_weights=sample_w[num]
        eig_val, eig_vec = sla.eigsh(lapLO, num_clusters+1, which='SM', tol=config.sc_eig_tol, maxiter=1000)
        obj = eig_val[num_clusters-1] / eig_val[num_clusters] - config.obj_alpha*eig_val[1]
        sample_obj.append(obj)

    # Parabolic interpolation
    x = np.asarray(sample_w)[:,:-1]
    y = np.asarray(sample_obj)
    poly_reg =PolynomialFeatures(degree=2) 
    X_ploy =poly_reg.fit_transform(x)
    lin_reg_2=linear_model.Ridge(alpha=config.ridge_alpha, fit_intercept=False)
    lin_reg_2.fit(X_ploy,y)
    
    w_constraint = [lambda w: 1.0 - np.sum(w), lambda w: w]
    def objective_function(w):
        view_weights[:-1] = w
        view_weights[-1] = 1.0 - np.sum(w)
        return lin_reg_2.predict(poly_reg.fit_transform(np.asarray(view_weights[:-1]).reshape(1,-1))) + config.obj_regular*np.power(np.asarray(view_weights),2).sum()
    opt_time=time.time()
    opt_w = fmin_cobyla(objective_function, np.full((nv-1), 1.0/nv), w_constraint, rhoend=config.opt_w_tol, maxfun=1000, rhobeg=config.opt_cobyla_rhobeg, catol=0.0000001, disp=3 if config.verbose else 0)
    if config.verbose:
        print(f"cobyla optimization time: {time.time()-opt_time}")
    view_weights[:-1] = opt_w
    view_weights[-1] = 1.0 - np.sum(opt_w)

    if config.embedding:
        delta=sp.eye(dataset['n'])-mv_lap(sp.eye(dataset['n']))
        if config.scale:
            from sketchne_graph import sketchne_graph
            emb = sketchne_graph(delta, dim = config.embed_dim, spec_propagation=False, window_size=10, eta1=32, eta2=32, eig_rank=config.embed_rank, power_iteration=20)
        else:
            from embedding import netmf
            emb = netmf(delta, dim = config.embed_dim,rank = config.embed_rank)
        embed_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ovr_evaluate(emb, dataset['labels'])
        print(f"Time: {embed_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")
    else: # clustering
        lapLO = sla.LinearOperator((n, n), matvec=mv_lap)
        try:
            eig_val, eig_vec = sla.eigsh(lapLO, num_clusters+2, which='SM', tol=config.sc_eig_tol, maxiter=1000)
        except sla.ArpackError as e:
            eig_val, eig_vec = e.eigenvalues, e.eigenvectors
            print(f"{' '.join([str(w) for w in view_weights])} {0.} {0.} {0.} {0.} {time.time() - start_time} {1.} NoConverge")
            return np.zeros(n)
        predict_clusters, _ = discretize(eig_vec[:, :num_clusters])
        cluster_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        cm = clustering_metrics(dataset['labels'], predict_clusters)
        acc, nmi, f1, _, ari, _ = cm.evaluationClusterModelFromLabel()
        print(f"Acc: {acc:.3f} F1: {f1:.3f} NMI: {nmi:.3f} ARI: {ari:.3f} Time: {cluster_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(config.dataset)
    if config.dataset.startswith("mag"):
        config.scale = True
    SMGF_PI(dataset)

