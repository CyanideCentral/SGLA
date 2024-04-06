
dataset = None
embedding = False

knn_k = 10
sc_eig_tol = 1e-2
optimize_weights = True # disable for UNIFORM variant
fixed_weights = [1./3] * 3
opt_cobyla_rhobeg = 0.05
opt_max_iters = 1000
opt_eig_tol = 0.01 # convergence threshold for eigensolver used for weight optimization
opt_w_tol = 1e-2 # convergence threshold for COBYLA optimizer
opt_objective = 'combine' # Optimization objective. 'con': connectivity, 'gap': eigengap, 'reg': relative eigen-gap
obj_alpha = 1.0 #connectivity obj weight alpha
obj_regular = 0.5 # regularizer weight gamma
seed = 0 #random seed
ridge_alpha = 0.05 #ridge linear regression
embed_dim = 64 # embedding demension
step_length = 0.5 #sample step for SMGF_PI funciton
scale = False # enable configurations for large-scale data
verbose = False # verbose output
embed_rank = 32 # NETMF/SKETCHNE parameter