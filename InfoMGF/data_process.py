import warnings
import sys, os

import scipy.sparse as sp
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import scipy

from sklearn.preprocessing import OneHotEncoder
from utils import normalize, symmetrize, remove_self_loop, sparse_tensor_add_self_loop, adj_values_one

warnings.simplefilter("ignore")
EOS = 1e-10


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    rowsum = features.sum(dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1,1)
    features = features * r_inv
    return features


class HIN(object):

    def __init__(self, dataset):
        data_path = f'data/{dataset}/'
        with open(f'{data_path}node_features.pkl', 'rb') as f:
            self.features = pickle.load(f)
        with open(f'{data_path}edges.pkl', 'rb') as f:
            self.edges = pickle.load(f)
        with open(f'{data_path}labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
        with open(f'{data_path}meta_data.pkl', 'rb') as f:
            self.__dict__.update(pickle.load(f))
        if scipy.sparse.issparse(self.features):
            self.features = self.features.todense()

    def to_torch(self):
        '''
        Returns the torch tensor of the graph.
        Returns:
            features, adj: feature and adj. matrix
            train_x, train_y, val_x, val_y, test_x, test_y: train/val/test index and labels
        '''
        features = torch.from_numpy(self.features).type(torch.FloatTensor)
        train_x, train_y, val_x, val_y, test_x, test_y = self.get_label()

        adj = np.sum(list(self.edges.values())).todense()
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        adj = F.normalize(adj, dim=1, p=2)

        return features, adj, train_x, train_y, val_x, val_y, test_x, test_y


    def get_label(self):
        '''
        Args:
            dev: device (cpu or gpu)

        Returns:
            train_x, train_y, val_x, val_y, test_x, test_y: train/val/test index and labels
        '''
        train_x = torch.from_numpy(np.array(self.labels[0])[:, 0]).type(torch.LongTensor)
        train_y = torch.from_numpy(np.array(self.labels[0])[:, 1]).type(torch.LongTensor)
        val_x = torch.from_numpy(np.array(self.labels[1])[:, 0]).type(torch.LongTensor)
        val_y = torch.from_numpy(np.array(self.labels[1])[:, 1]).type(torch.LongTensor)
        test_x = torch.from_numpy(np.array(self.labels[2])[:, 0]).type(torch.LongTensor)
        test_y = torch.from_numpy(np.array(self.labels[2])[:, 1]).type(torch.LongTensor)
        return train_x, train_y, val_x, val_y, test_x, test_y


def data_processed(dataset):

    g = HIN(dataset)
    target_index = g.t_info[g.types[0]]['ind']

    features, adj_all, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch()
    features_target = features[target_index]

    nnodes = len(features_target)
    all_labels = torch.zeros(nnodes, dtype=torch.int64)

    for i, idx in enumerate(train_x):
        all_labels[idx] = train_y[i]
    for i, idx in enumerate(val_x):
        all_labels[idx] = val_y[i]
    for i, idx in enumerate(test_x):
        all_labels[idx] = test_y[i]

    nclasses = int(all_labels.max() + 1)

    train_mask = torch.zeros(nnodes, dtype=torch.int64)
    train_mask[train_x] = 1
    val_mask = torch.zeros(nnodes, dtype=torch.int64)
    val_mask[val_x] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.int64)
    test_mask[test_x] = 1

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()

    if dataset == 'acm':
        adjs = []
        ri = g.r_info
        for r in ['p-a', 'p-s']:
            adj_ = adj_all[ri[r][0]:ri[r][1], ri[r][2]:ri[r][3]]
            adj = adj_ @ adj_.t()
            adjs.append(adj)

    elif dataset == 'yelp':
        adjs = []
        ri = g.r_info
        for r in ['b-u', 'b-s', 'b-l']:
            adj_ = adj_all[ri[r][0]:ri[r][1], ri[r][2]:ri[r][3]]
            adj = adj_ @ adj_.t()
            adjs.append(adj)

    elif dataset == 'dblp':
        adjs = []
        ri = g.r_info

        adj_ap = adj_all[ri['a-p'][0]:ri['a-p'][1], ri['a-p'][2]:ri['a-p'][3]]
        adj_apa = adj_ap @ adj_ap.t()
        adjs.append(adj_apa)

        adj_pc = adj_all[ri['p-c'][0]:ri['p-c'][1], ri['p-c'][2]:ri['p-c'][3]]
        adj_apc = adj_ap @ adj_pc
        adj_apcpa = adj_apc @ adj_apc.t()
        adjs.append(adj_apcpa)


    adjs = [symmetrize(adj).to_sparse() for adj in adjs]
    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return features_target, features_target.shape[1], all_labels, nclasses, train_mask, val_mask, test_mask, adjs


features, _, labels, _, train_mask, val_mask, test_mask, adjs = data_processed('acm')
path = "./data/acm/processed/"

torch.save(features,path+'features.pt')
torch.save(labels,path + "label.pt")
torch.save(adjs,path+'adj.pt')

torch.save(train_mask,path+'train_mask.pt')
torch.save(val_mask,path+'val_mask.pt')
torch.save(test_mask,path+'test_mask.pt')
