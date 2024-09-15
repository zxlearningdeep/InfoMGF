import torch
import torch.nn as nn
from graph_learners import *
from utils import *
import copy
from params import *
from model import calc_upper_bound, sce_loss


def graph_augment(adjs_original, dropedge_rate, training, sparse):

    adjs_aug = []
    adjs = copy.deepcopy(adjs_original)

    if not sparse:
        adjs = [adj.to_sparse().coalesce() for adj in adjs]
    for i in range(len(adjs)):
        adj_aug = adjs[i]
        diag_indices, diag_value = get_sparse_diag(adj_aug)
        adj_aug = remove_self_loop([adj_aug])[0]
        value = F.dropout(adj_aug.values(), p=dropedge_rate, training=training)

        adj_aug = torch.sparse.FloatTensor(torch.cat((adj_aug.indices(), diag_indices), dim=1), torch.cat((value, diag_value), dim=0), adj_aug.shape).coalesce().to(adjs[i].device)
        adjs_aug.append(adj_aug)
    if sparse:
        adjs_aug = [torch_sparse_to_dgl_graph(a) for a in adjs_aug]
    else:
        adjs_aug = [adj.to_dense() for adj in adjs_aug]

    return adjs_aug


def graph_generative_augment(adjs, features, discriminator, sparse):

    adjs_aug = []
    if not sparse:
        adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)
    for i in range(len(adjs)):
        edge_index = adjs[i].indices()
        adj_aug_value = discriminator(features, edge_index)
        adj_aug = torch.sparse.FloatTensor(edge_index, adj_aug_value, adjs[i].shape).to(adjs[i].device)

        adj_aug = sparse_tensor_add_self_loop(adj_aug)
        adj_aug = normalize(adj_aug, 'sym', sparse=True)

        adjs_aug.append(adj_aug)

    if sparse:
        adjs_aug = [torch_sparse_to_dgl_graph(a) for a in adjs_aug]
    else:
        adjs_aug = [adj.to_dense() for adj in adjs_aug]
    return adjs_aug


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, rep_dim, aug_lambda, temperature=1.0, bias=0.0 + 0.0001):
        super(Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, 1))

        self.temperature = temperature
        self.bias = bias
        self.aug_lambda = aug_lambda

        self.decoder = nn.Sequential(nn.Linear(rep_dim, input_dim))

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h

    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2

    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(edges_weights_raw.device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        output = torch.sigmoid(gate_inputs).squeeze()

        return output

    def forward(self, embedding, edges):
        embedding_ = self.get_node_embedding(embedding)
        edges_weights_raw = self.get_edge_weight(embedding_, edges)
        weights = self.gumbel_sampling(edges_weights_raw)
        return weights

    def cal_loss_dis(self, z_aug_adjs, z_specific_adjs, view_features):

        loss_upmi = 0
        loss_rec = 0
        batch_size, _ = z_specific_adjs[0].size()
        pos_eye = torch.eye(batch_size).to(z_specific_adjs[0].device)
        for i in range(len(z_specific_adjs)):
            loss_upmi += self.aug_lambda * calc_upper_bound(z_aug_adjs[i], z_specific_adjs[i], pos_eye)

            feat_agg_rec = self.decoder(z_aug_adjs[i])
            loss_rec += sce_loss(feat_agg_rec, view_features[i])

        loss_dis = loss_upmi + loss_rec
        return loss_dis


