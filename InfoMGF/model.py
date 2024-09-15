import copy
import math
import dgl
import torch

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl
from torch.nn import Sequential, Linear, ReLU

# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))
        self.dropout = dropout
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.sparse = sparse

    def forward(self, x):
        Adj = copy.deepcopy(self.Adj)
        if self.sparse:
            Adj.edata['w'] = self.dropout_adj(Adj.edata['w'])
        else:
            Adj = self.dropout_adj(Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()

        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))
        self.sparse = sparse

    def forward(self, x, Adj):

        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn_encoder_layers[-1](x, Adj)
        return x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, num_g):
        super(GCL, self).__init__()

        self.num_g = num_g
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse)

        self.proj_s = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])

        self.proj_u = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])

        self.proj_f = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])


    def forward(self, x, Adj_):
        embedding = self.encoder(x, Adj_)
        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding

    def cal_loss(self, z_specific_adjs, z_aug_adjs, z_fused_adj):

        batch_size, _ = z_fused_adj.size()
        pos_eye = torch.eye(batch_size).to(z_fused_adj.device)
        z_proj_s = [self.proj_s[i](z_specific_adjs[i]) for i in range(self.num_g)]
        z_proj_u = [self.proj_u[i](z_aug_adjs[i]) for i in range(self.num_g)]

        loss_smi = 0
        cnt = 0
        for i in range(self.num_g):
            for j in range(i+1, self.num_g):
                loss_smi += calc_lower_bound(z_proj_s[i], z_proj_s[j], pos_eye)
                cnt += 1
        loss_smi = loss_smi / cnt

        loss_fused = 0
        loss_umi = 0

        for i in range(self.num_g):
            loss_fused += calc_lower_bound(self.proj_f[i](z_fused_adj), z_proj_s[i], pos_eye)
            loss_umi += calc_lower_bound(z_proj_s[i], z_proj_u[i], pos_eye)

        loss_fused = loss_fused / self.num_g
        loss_umi = loss_umi / self.num_g
        loss = loss_fused + loss_smi + loss_umi
        return loss




def AGG(h_list, adjs_o, nlayer, sparse=False):
    f_list = []
    for i in range(len(adjs_o)):
        z = h_list[i]
        adj = adjs_o[i]
        for i in range(nlayer):
            if sparse:
                z = torch.sparse.mm(adj, z)
            else:
                z = torch.matmul(adj, z)
        z = F.normalize(z, dim=1, p=2)
        f_list.append(z)

    return f_list



def sim_con(z1, z2, temperature):
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EOS
    sim_matrix = dot_numerator / dot_denominator / temperature
    return sim_matrix


def calc_lower_bound(z_1, z_2, pos, temperature=0.2):
    matrix_1 = torch.exp(sim_con(z_1, z_2, temperature))
    matrix_2 = matrix_1.t()

    matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EOS)
    lori_1 = -torch.log(matrix_1.mul(pos).sum(dim=-1)).mean()

    matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EOS)
    lori_2 = -torch.log(matrix_2.mul(pos).sum(dim=-1)).mean()

    return (lori_1 + lori_2) / 2


def calc_upper_bound(z_1, z_2, pos, temperature=0.2):
    matrix_1 = sim_con(z_1, z_2, temperature)
    loss = matrix_1.mul(pos).sum(dim=-1).mean() - matrix_1.mean()

    return loss


def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()
    return loss


