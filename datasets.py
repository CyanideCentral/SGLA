import pickle
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import config as config

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def load_data(dataset_name):
    dataset = {}
    # Six multiplex datasets
    if dataset_name == 'acm':
        data = sio.loadmat("./data/acm/ACM3025.mat")
        dataset['graphs'] = [sp.csr_matrix(data['PAP']), sp.csr_matrix(data['PLP'])]
        dataset['features'] = [sp.csr_matrix(data['feature'])]
        dataset['labels'] = data['label']
    elif dataset_name == 'dblp':
        data = sio.loadmat("./data/dblp/DBLP4057_GAT_with_idx.mat")
        dataset['graphs'] = [sp.csr_matrix(data['net_APA']), sp.csr_matrix(data['net_APCPA']), sp.csr_matrix(data['net_APTPA'])]
        dataset['features'] = [sp.csr_matrix(data['features'])]
        dataset['labels'] = data['label']
    elif dataset_name == 'imdb':
        data = pickle.load(open("./data/imdb/imdb.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['MAM'], data['MDM']]]
        dataset['features'] = [data['feature']]
        dataset['labels'] = data['label']
    elif dataset_name == 'yelp':
        data = pickle.load(open("./data/yelp/yelp.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['BUB'], data['BSB']]]
        dataset['features'] = [data['features']]
        dataset['labels'] = data['labels']
    elif dataset_name == 'freebase':
        data = pickle.load(open("./data/freebase/freebase.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['MAM'], data['MWM'],data['MDM']]]
        dataset['features'] = [data['feature']]
        dataset['labels'] = data['label']
    elif dataset_name =='rm':
        data = pickle.load(open(f"./data/ftcs/rm.pkl", "rb"))
        dataset['graphs'] = data['graphs']
        dataset['features'] = data['features']
        dataset['labels'] = data['labels']
    # Amazon datasets consisting of one graph and multiple features
    elif dataset_name =='query':
        data = pickle.load(open("./data/query/query.pkl", "rb"))
        dataset['graphs'] = [sp.csr_matrix(g) for g in [data['graphs']]]
        dataset['features'] = [data['features'],data['features1']]
        dataset['labels'] = data['label']
    elif dataset_name in ['amazon-computers', 'amazon-photos']:
        mcgc_file_names = {'amazon-computers': 'amazon_electronics_computers.npz', 'amazon-photos': 'amazon_electronics_photo.npz'}
        data = dict(np.load(f"./data/mcgc/{mcgc_file_names[dataset_name]}"))
        dataset['graphs'] = [sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])]
        attr_matrix = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape']).toarray()
        dataset['features'] = [attr_matrix,attr_matrix@attr_matrix.T]
        dataset['labels'] = data['labels']
    # MAG datasets consisting of multiple graphs and features
    elif dataset_name in ['magphy','mageng']:
        data = pickle.load(open(f"./data/scale/new_{dataset_name}.pkl", "rb"))
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
        # dataset['features'][i] = preprocess_features(dataset['features'][i])
        dataset['n'].append(dataset['features'][i].shape[0])
    dataset['nv'] = len(dataset['n']) # number of views

    dataset['labels'] = np.asarray(np.argmax(dataset['labels'], axis=1)).flatten() if dataset['labels'].ndim == 2 else dataset['labels']
    if dataset['labels'].min()==1:
        dataset['labels'] -= 1
    dataset['k'] = len(np.unique(dataset['labels']))
    dataset['labels_nk'] = np.eye(dataset['k'])[dataset['labels']]
    dataset['n'].append(dataset['labels'].shape[0])
    if np.unique(dataset['n']).shape[0] > 1:
        raise Exception('Inconsistent number of nodes')
    dataset['n'] = dataset['n'][0]
    return dataset
