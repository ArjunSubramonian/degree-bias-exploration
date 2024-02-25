# +
# Copyright (c) 2023 PyG Team <team@pyg.org>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor
)
from torch_geometric.utils import add_remaining_self_loops
import torch_scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)

def scatter(src: Tensor, index: Tensor, dim: int = 0,
                dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:

    if reduce == 'any':
        dim = src.dim() + dim if dim < 0 else dim

        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_(dim, index, src)

    return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                 reduce=reduce)

# +
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, flow="source_to_target", conv="sym", dtype=None):

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

#     edge_index, edge_weight = add_remaining_self_loops(
#             edge_index, edge_weight, 1.0, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    
    if conv == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif conv == "rw":
        deg_inv = deg.pow_(-1.0)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]

    return edge_index, edge_weight


# -

class GCNConv(MessagePassing):

    def __init__(self, feat_channels: int, in_channels: int, out_channels: int, conv: str, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv

        self.lin_self = Linear(in_channels, out_channels, bias=True,
                          weight_initializer='glorot')
        self.lin = Linear(in_channels, out_channels, bias=True,
                          weight_initializer='glorot')
        self.lin_res = Linear(feat_channels, out_channels, bias=True,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()


    def forward(self, x: Tensor, x0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(self.node_dim), self.flow, self.conv, x.dtype)
         
        x_self = self.lin_self(x)
        x = x_self + self.lin(x) + self.lin_res(x0)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
