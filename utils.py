# +
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from torch_geometric.utils import degree, add_remaining_self_loops, k_hop_subgraph, to_networkx, subgraph
from torch_scatter import scatter
import torch_sparse
from torch_sparse import coalesce
import networkx as nx

def get_id(n, edge_index):
    index = torch.tensor([list(range(n)), list(range(n))], device=edge_index.device).long()
    value = torch.ones_like(index[0]).float()
    return coalesce(index, value, m=n, n=n)

def power(edge_index, edge_weight, n, k):
    if k == 0:
        pow_edge_index, pow_edge_weight = get_id(n, edge_index)
        return pow_edge_index, pow_edge_weight
        
    pow_edge_index = edge_index.clone()
    pow_edge_weight = edge_weight.clone()
    for _ in range(k - 1):
        pow_edge_index, pow_edge_weight = torch_sparse.spspmm(edge_index, edge_weight, pow_edge_index, pow_edge_weight, n, n, n)
    return pow_edge_index, pow_edge_weight

def power_x(edge_index, edge_weight, x, n, k):
    for _ in range(k):
        x = torch_sparse.spmm(edge_index, edge_weight, n, n, x)
    return x

def gcn_norm(edge_index, edge_weight, n, return_inv_sqrt=False):
    deg = degree(edge_index[0], num_nodes=n)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    if return_inv_sqrt:
        return deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]], deg_inv_sqrt
    return deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]

def sage_norm(edge_index, edge_weight, n):
    deg = degree(edge_index[0], num_nodes=n)
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    return edge_weight * deg_inv[edge_index[1]]


# +
@torch.no_grad()
def test_err_by_deg(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_weight = torch.ones_like(edge_index[0]).float()
    edge_weight = gcn_norm(edge_index, edge_weight, data.x.size(0))
    
    deg = torch.zeros_like(data.y).float()
    deg = torch.scatter_add(deg, 0, edge_index[0], edge_weight)

    errs = []
    degs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        mismatches = (pred[mask] != data.y[mask]).float() 
        
        errs.append(mismatches)
        degs.append(deg[mask])
    return errs, degs

@torch.no_grad()
def test_loss_by_deg(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_weight = torch.ones_like(edge_index[0]).float()
    edge_weight = gcn_norm(edge_index, edge_weight, data.x.size(0))
    
    deg = torch.zeros_like(data.y).float()
    deg = torch.scatter_add(deg, 0, edge_index[0], edge_weight)

    errs = []
    degs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = F.log_softmax(out[mask], dim=1)
        target = data.y[mask]
        err = -pred[range(target.size(0)), target]
         
        errs.append(err)
        degs.append(deg[mask])
    return errs, degs


# -

def label_homo(data, k):
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_weight = torch.ones_like(edge_index[0]).float()
    edge_weight = gcn_norm(edge_index, edge_weight, data.x.size(0))
    
    deg = torch.zeros_like(data.y).float()
    deg = torch.scatter_add(deg, 0, edge_index[0], edge_weight)
    
    homo = torch.zeros_like(data.y).float()
    pow_edge_index, pow_edge_weight = power(edge_index, edge_weight, data.x.size(0), k)
    c_mask = data.y[pow_edge_index[0]] == data.y[pow_edge_index[1]]
    homo = torch.scatter_add(homo, 0, pow_edge_index[0, c_mask], pow_edge_weight[c_mask]) / deg
    
    return [homo[mask] for mask in [data.train_mask, data.val_mask, data.test_mask]]


def feature_homo(data, k):
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_weight = torch.ones_like(edge_index[0]).float()
    edge_weight = gcn_norm(edge_index, edge_weight, data.x.size(0))
    
    deg = torch.zeros_like(data.y).float()
    deg = torch.scatter_add(deg, 0, edge_index[0], edge_weight)
    
    homo = torch.zeros_like(data.y).float()
    pow_edge_index, pow_edge_weight = power(edge_index, edge_weight, data.x.size(0), k)
    
    x_lowrank = torch.svd(data.x)[0][:, :2]
    dists = torch.norm(x_lowrank[pow_edge_index[0]] - x_lowrank[pow_edge_index[1]], dim=1)
    homo = torch.scatter_add(homo, 0, pow_edge_index[0], dists * pow_edge_weight)
    
    return [homo[mask] for mask in [data.train_mask, data.val_mask, data.test_mask]]


def influence_scores(data, k):
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    edge_weight = torch.ones_like(edge_index[0]).float()
    edge_weight = gcn_norm(edge_index, edge_weight, data.x.size(0))
    
    edge_index, edge_weight = power(edge_index, edge_weight, data.x.size(0), k)
    
    deg = torch.zeros_like(data.y).float()
    deg = torch.scatter_add(deg, 0, edge_index[0], edge_weight)
    
    labeled_idx = data.train_mask.float().reshape(-1, 1)
    labeled_influence = torch_sparse.spmm(edge_index, edge_weight, data.x.size(0), data.x.size(0), labeled_idx).flatten()
    self_influence = edge_weight[edge_index[0] == edge_index[1]] 
    
    return [labeled_influence[mask] / deg[mask] for mask in [data.train_mask, data.val_mask, data.test_mask]]


def get_compatibility_matrix(y, edge_index, edge_weight=None):
    """
    Return the weighted compatibility matrix, according to the weights in the provided adjacency matrix.
    """
    src, dst = edge_index

    num_classes = torch.unique(y).shape[0]
    H = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            mask = (y == i)[src] & (y == j)[dst]
            H[i, j] = edge_weight[mask].sum()

    return torch.nn.functional.normalize(H, p=1)


# +
def scatter(x, high_degs_mask, low_degs_mask, data):
    low_deg_spread = 0
    high_deg_spread = 0
    for c in range(data.y.max() + 1):
        mask = low_degs_mask & (data.y[data.test_mask] == c)
        if mask.sum() != 0:
            centered_x = x[mask] - x[mask].mean(dim=0).reshape(1, -1)
            low_deg_spread += centered_x.t() @ centered_x / low_degs_mask.sum()
        
        mask = high_degs_mask & (data.y[data.test_mask] == c)
        if mask.sum() != 0:
            centered_x = x[mask] - x[mask].mean(dim=0).reshape(1, -1)
            high_deg_spread += centered_x.t() @ centered_x / high_degs_mask.sum()
    return torch.trace(low_deg_spread), torch.trace(high_deg_spread)

#     low_deg_spread = 0
#     high_deg_spread = 0
    
#     mask = low_degs_mask
#     centered_x = x[mask] - x[mask].mean(dim=0).reshape(1, -1)
#     low_deg_spread += centered_x.t() @ centered_x / low_degs_mask.sum()
        
#     mask = high_degs_mask
#     centered_x = x[mask] - x[mask].mean(dim=0).reshape(1, -1)
#     high_deg_spread += centered_x.t() @ centered_x / high_degs_mask.sum()
    
#     return torch.trace(low_deg_spread), torch.trace(high_deg_spread)
