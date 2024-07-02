import pickle
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import config as config

def load_data(dataset_name):
    dataset = {}
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        data = sio.loadmat("./data/DBLP4057_GAT_with_idx.mat")
        dataset['graphs'] = [sp.csr_matrix(data['net_APA']), sp.csr_matrix(data['net_APCPA']), sp.csr_matrix(data['net_APTPA'])]
        dataset['features'] = [sp.csr_matrix(data['features'])]
        dataset['labels'] = data['label']
    elif dataset_name == 'imdb':
        data = pickle.load(open("./data/imdb.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['MAM'], data['MDM']]]
        dataset['features'] = [data['feature']]
        dataset['labels'] = data['label']
    elif dataset_name == 'yelp':
        data = pickle.load(open("./data/yelp.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['BUB'], data['BSB']]]
        dataset['features'] = [data['features']]
        dataset['labels'] = data['labels']
    elif dataset_name =='rm':
        data = pickle.load(open(f"./data/rm.pkl", "rb"))
        dataset['graphs'] = data['graphs']
        dataset['features'] = data['features']
        dataset['labels'] = data['labels']
    elif dataset_name in ['amazon-computers', 'amazon-photos']:
        data = dict(np.load(f"./data/{dataset_name}.npz"))
        dataset['graphs'] = [sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])]
        attr_matrix = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape']).toarray()
        dataset['features'] = [attr_matrix,attr_matrix@attr_matrix.T]
        dataset['labels'] = data['labels']
    elif dataset_name in ['magphy','mageng']:
        data = pickle.load(open(f"./data/{dataset_name}.pkl", "rb"))
        dataset['graphs'] = [data['PP'], data['AP'].T@data['AP']]
        dataset['features'] = [data['features'],data['features1']]
        dataset['labels'] = data['labels']
    else:
        raise Exception('Invalid dataset name')

    # data cleanup
    dataset['n'] = []
    for i in range(len(dataset['graphs'])):
        # remove self loops
        if sp.issparse(dataset['graphs'][i]):
            dataset['graphs'][i].setdiag(0)
        else:
            np.fill_diagonal(dataset['graphs'][i], 0)
        # convert to undirected graph, if necessary
        if (dataset['graphs'][i]!=dataset['graphs'][i].T).sum() > 0:
            dataset['graphs'][i] = dataset['graphs'][i] + dataset['graphs'][i].T
        dataset['graphs'][i][dataset['graphs'][i]>1] = 1
        if sp.issparse(dataset['graphs'][i]):
            dataset['graphs'][i].eliminate_zeros()
        dataset['n'].append(dataset['graphs'][i].shape[0])
    for i in range(len(dataset['features'])):
        dataset['n'].append(dataset['features'][i].shape[0])
    dataset['nv'] = len(dataset['n']) # number of views

    dataset['labels'] = np.asarray(np.argmax(dataset['labels'], axis=1)).flatten() if dataset['labels'].ndim == 2 else dataset['labels']
    if dataset['labels'].min()==1:
        dataset['labels'] -= 1
    dataset['k'] = len(np.unique(dataset['labels']))
    dataset['n'].append(dataset['labels'].shape[0])
    if np.unique(dataset['n']).shape[0] > 1:
        raise Exception('Inconsistent number of nodes')
    dataset['n'] = dataset['n'][0]
    return dataset
