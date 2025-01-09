from typing import Tuple
import math
import torch
import torch.nn as nn
from torch_geometric.nn import GIN, MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

from models.model_utils import index_select_ND


class GINConv(MessagePassing):
    def __init__(self, emb_dim, bond_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        # use mlp instead of embedding
        self.w_bond = nn.Sequential(
            nn.Linear(bond_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim))
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges for original edge.
        edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.size(0), 1), device=edge_attr.device)), dim=-1)
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)
        self_loop_attr[:, -1] = 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        edge_embeddings = self.w_bond(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINE(nn.Module):
    def __init__(self, hidden_size, depth, atom_message, atom_fdim, bond_fdim, dropout):
        super(GINE, self).__init__()
        self.hidden_size = hidden_size
        self.atom_message = atom_message
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.depth = depth
        self.dropout = dropout
        input_dim = self.atom_fdim
        self.gins = torch.nn.ModuleList()
        for layer in range(depth):
            self.gins.append(GINConv(hidden_size, bond_fdim, aggr="add"))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.w_atom = nn.Linear(input_dim, self.hidden_size, bias=False)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(depth):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_size))
        self.Dropout = nn.Dropout(p=self.dropout)
        self.JK = 'last'
        # self.W_g = nn.Sequential(
        #     nn.Linear(input_size + hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size))

        # self.output_mess = nn.Sequential(
        #     nn.Linear(depth * hidden_size, int(depth / 2) * hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(int(depth / 2) * hidden_size, hidden_size))

    def forward(self, graph_tensors):
        # a2b -> atom to incoming bond  b2a -> bond to atom coming from
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        pos = torch.nonzero(a2b > 0)
        to = torch.zeros(b2a.shape, dtype=torch.long, device=b2a.device)
        to[a2b[pos[:, 0], pos[:, 1]]] = pos[:, 0]  # edgeidx(a2b[...]) -> atom_idx
        # (2,edge) coming from , going to
        edge_index = torch.stack((b2a, to))
        h_atom = self.w_atom(f_atoms)
        h_atom = [h_atom]
        for layer in range(self.depth):
            h = self.gins[layer](h_atom[layer], edge_index, f_bonds)
            h = self.batch_norms[layer](h)
            if layer == self.depth - 1:
                h = self.Dropout(h)
            else:
                h = self.Dropout(F.relu(h))
            h_atom.append(h)
        if self.JK == "concat":
            node_representation = torch.cat(h_atom, dim=1)
        elif self.JK == "last":
            node_representation = h_atom[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_atom]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_atom]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        return node_representation
