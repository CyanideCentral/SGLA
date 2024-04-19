import scipy.sparse as sp
import scipy.linalg as sla
import numpy as np
import time
from sparse_dot_mkl import dot_product_mkl

def eigSVD(X):
    M = X.T @ X
    S, V = np.linalg.eigh(M)
    eigvals_sqrt = np.sqrt(S)
    if eigvals_sqrt[0] == 0:
        raise ValueError("Eigenvalue 0 encountered in eigSVD")
    eigvals_sqrt_inv = 1/eigvals_sqrt
    Q = X @ (V * eigvals_sqrt_inv)
    return Q

def deepwalk_filter(eigvals, window_size):
    for i in range(eigvals.size):
        x = eigvals[i]
        if x >= 1:
            eigvals[i] = 1
        else:
            eigvals[i] = x * (1 - x**window_size) / (1 - x) / window_size
        eigvals[i] = max(eigvals[i], 0)
    return eigvals

def freigs(A, eig_rank, power_iteration, oversampling, convex_projection, upper, use_qr=False):
    l = eig_rank + oversampling
    if upper:
        A_triu = sp.triu(A)
        A = A_triu + A_triu.T - sp.diags(A.diagonal())
    omega = np.random.default_rng().random((A.shape[1], l))
    Y = dot_product_mkl(A, omega, cast=True)
    if use_qr:
        Q, _ = sla.qr(Y, mode='economic')
    else:
        Q = eigSVD(Y)
    pi_start = time.time()
    eigsvd_time = 0.
    for i in range(power_iteration):
        Q1 = dot_product_mkl(A, Q, cast=True)
        Q2 = dot_product_mkl(A, Q1, cast=True)
        eigsvd_time_start = time.time()
        Q = eigSVD(Q2)
        eigsvd_time += time.time() - eigsvd_time_start
    print(f"Power iteration time: {time.time()-pi_start}")
    print(f"eigSVD time: {eigsvd_time}")
    if convex_projection: 
        pass
    else:
        M = Q.T @ dot_product_mkl(A, Q, cast=True)
        SS, MV = np.linalg.eigh(M, UPLO='U')
        UU2 = MV[:, -eig_rank:]
        U = Q @ UU2
        S = SS[-eig_rank:]
    return S, U

def random_signs(length):
    return np.random.choice([-1, 1], length)

def sketch_svds(F, CF, dim, s1, s2, eta1, eta2, normalize):
    n, eig_rank = F.shape
    l1 = dim + s1
    l2 = dim + s2
    spMat1 = sp.random(n, l1, density=(eta1*l1+0.5)/n/l1, format='csc', dtype=np.float32, data_rvs=random_signs, random_state=np.random.default_rng())
    CF_compressed = CF[:, spMat1.indices]
    M_compressed = F @ CF_compressed
    M_compressed += 1.0
    M_compressed.clip(min=1.0, out=M_compressed)
    np.log(M_compressed, out=M_compressed)
    Y = dot_product_mkl(M_compressed, spMat1[spMat1.indices,:],cast=True)
    Q = eigSVD(Y)

    spMat2 = sp.random(n, l2, density=(eta2*l2+0.5)/n/l2, format='csc', dtype=np.float32, data_rvs=random_signs, random_state=np.random.default_rng())
    F_compressed = F[spMat2.indices, :]
    CF_compressed = CF[:, spMat2.indices]
    M_core = F_compressed @ CF_compressed
    spMat2_core = spMat2[spMat2.indices,:]
    M_right = dot_product_mkl(M_core, spMat2_core,cast=True)
    Z = dot_product_mkl(spMat2_core.T, M_right, cast=True)
    temp = dot_product_mkl(spMat2.T, Q, cast=True)
    Utemp, Rtemp = np.linalg.qr(temp)
    UtZ = Utemp.T @ Z
    UtZ = np.linalg.solve(Rtemp, UtZ)
    UtTt = Utemp.T @ UtZ.T
    UtTt = np.linalg.solve(Rtemp, UtTt)
    Z = UtTt.T
    Uc, Sc, Vc = np.linalg.svd(Z)
    Uk = Uc[:, :dim]
    U = Q @ Uk
    S = Sc[:dim]
    emb = U * np.sqrt(S)
    if normalize:
        emb_norm = np.linalg.norm(emb, axis=1)
        emb_norm[emb_norm==0] = 1
        emb /= emb_norm[:, np.newaxis]
    return emb

def sketchne_graph(adj_matrix, window_size=10, negative_samples=1, alpha=0.5, eig_rank=256, power_iteration=10, oversampling=20, convex_projection=False, dim=128, eta1=8, eta2=8, s1=100, s2=1000, normalize=True, order=10, theta=0.5, mu=0.2, upper=False):
    n = adj_matrix.shape[0]
    m = (adj_matrix.nnz - (adj_matrix.diagonal()>0).sum())/2
    deg_vec = adj_matrix.sum(axis=1).A1
    deg_vec[deg_vec==0] = 1
    deg_alpha = deg_vec**(-alpha)
    D_alpha = sp.diags(deg_alpha, format='csr')
    L = dot_product_mkl(D_alpha, dot_product_mkl(adj_matrix, D_alpha))

    start_time = time.perf_counter()
    eigvals, U = freigs(L, eig_rank, power_iteration, oversampling, convex_projection, upper)
    print(f"freigs time: {time.perf_counter()-start_time}")

    if alpha == 0.5:
        eigvals = deepwalk_filter(eigvals, window_size)
        para = m / negative_samples
        eigvals *= para
        F = dot_product_mkl(D_alpha, U)
        CF = dot_product_mkl(sp.diags(eigvals, format='csr'), F.T)
    else:
        E_eigvals = sp.diags(eigvals, format='csr')
        F = dot_product_mkl(sp.diags(deg_vec ** (-1+alpha), format='csr'), U)
        res = dot_product_mkl(sp.diags(deg_vec ** (-1+2*alpha), format='csr'), U)
        Ki = U.T @ res
        K = dot_product_mkl(Ki , E_eigvals)
        K_iter = sp.identity(eig_rank, format='csr')
        evalsm = sp.identity(eig_rank, format='csr')
        for i in range(window_size-1):
            Ki = dot_product_mkl(K_iter, K)
            K_iter = Ki
            evalsm += K_iter
        evalsm = dot_product_mkl(E_eigvals, evalsm)
        para = m / negative_samples / window_size
        evalsm *= para
        CF = evalsm @ F.T
    
    emb = sketch_svds(F, CF, dim, s1, s2, eta1, eta2, normalize)
    return emb
