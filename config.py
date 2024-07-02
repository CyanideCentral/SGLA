
scale = False # enable configurations for large-scale data
verbose = False # verbose output
knn_k = 10 # K for KNN graph construction
eig_tol = 0.01 # precision of eigsh solver
opt_cobyla_rhobeg = 0.05 # COBYLA step size parameter
opt_t_max = 100 # maximum number of iterations for COBYLA optimizer
opt_epsilon = 0.001 # convergence threshold for COBYLA optimizer
obj_gamma = 0.5 # coefficient of weight regularization
ridge_alpha = 0.05 # regularization parameter for ridge regression

# embedding task
embedding = False # True for embedding task; False for clustering task
embed_dim = 64 # embedding output demension