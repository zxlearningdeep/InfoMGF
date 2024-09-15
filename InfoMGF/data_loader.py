import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import scipy
from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder
from utils import sparse_mx_to_torch_sparse_tensor, normalize, symmetrize, remove_self_loop, sparse_tensor_add_self_loop, adj_values_one

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


def load_mag():

    path = "./data/mag-4/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)

    feat = torch.load(path+'feat.pt').float()
    nnodes = len(feat)
    adj_1 = torch.load(path+'pap.pt').coalesce()
    adj_2 = torch.load(path+'pp.pt').coalesce()
    adjs = [adj_1, adj_2]
    adjs = [sparse_tensor_add_self_loop(adj).coalesce() for adj in adjs]
    adjs = [adj_values_one(adj) for adj in adjs]
    adjs = [normalize(adj, mode='sym', sparse=True) for adj in adjs]

    train_index = torch.load(path+'train_index.pt')
    val_index = torch.load(path+'val_index.pt')
    test_index = torch.load(path+'test_index.pt')

    train_mask = torch.zeros(nnodes, dtype=torch.int64)
    train_mask[train_index] = 1
    val_mask = torch.zeros(nnodes, dtype=torch.int64)
    val_mask[val_index] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.int64)
    test_mask[test_index] = 1

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_dblp():
    path = "data/dblp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')


    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_acm():
    path = "./data/acm/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_yelp():
    path = "./data/yelp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = F.normalize(feat, p=2, dim=1)
    adjs = torch.load(path+'adj.pt')

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    adjs = [normalize(adj, mode='sym') for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs


def load_data(args):
    if args.dataset == 'dblp':
        return load_dblp()
    elif args.dataset == 'mag':
        return load_mag()
    elif args.dataset == 'acm':
        return load_acm()
    elif args.dataset == 'yelp':
        return load_yelp()