from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import index_select_ND


class GRUMPNN(nn.Module):
    def __init__(self, hidden_size, depth, atom_message, atom_fdim, bond_fdim, dropout):
        super(GRUMPNN, self).__init__()
        self.hidden_size = hidden_size
        self.atom_message = atom_message
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.depth = depth
        self.dropout = dropout
        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.W_o = nn.Sequential(
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU())
        # self.W_g = nn.Sequential(
        #     nn.Linear(input_size + hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size))

        # self.output_mess = nn.Sequential(
        #     nn.Linear(depth * hidden_size, int(depth / 2) * hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(int(depth / 2) * hidden_size, hidden_size))

    def forward(self, graph_tensors, edit_embedding=None):
        if len(graph_tensors) == 6:
            f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        elif len(graph_tensors) == 4:
            f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        if self.atom_message:
            a2a = b2a[a2b]  # num_atoms x max_num_bonds
            f_bonds = f_bonds[:, -self.bond_fdim:]
            input = self.w_i(f_atoms)  # num_atoms x hidden
        else:
            input = self.w_i(f_bonds)  # num_bonds x hidden
        message = input
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding
        # mpnn层数
        for depth in range(self.depth - 1):
            if self.atom_message:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2a)
                # num_atoms x max_num_bonds x bond_fdim
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                # num_atoms x max_num_bonds x hidden + bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                # num_atoms x hidden + bond_fdim
                message = nei_message.sum(dim=1)
                message = self.w_h(message)  # num_bonds x hidden
            else:
                # num_atoms x max_num_bonds x hidden  message -> hidden
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden sum(h_ki)
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
                # a_message[b2a] = sum(ki) - hij的对称边
            message = self.gru(input, message)  # num_bonds x hidden_size U function
            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden
        if self.atom_message:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2a)
        else:
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden
        return atom_hiddens


class GRULight(nn.Module):
    def __init__(self, hidden_size, depth, node_f_dim, edge_f_dim, dropout, vocab_size, edge_init_dim):
        super(GRULight, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.node_dim = node_f_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.edit_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=node_f_dim)
        self.edge_dim = edge_init_dim
        self.w_edge = nn.Linear(self.edge_dim, edge_f_dim, bias=False)
        # self.w_h = nn.Linear(edge_f_dim + node_f_dim, node_f_dim)
        self.gru = nn.GRUCell(node_f_dim, node_f_dim)
        self.read_out = nn.Sequential(nn.Linear(node_f_dim + node_f_dim, hidden_size), nn.ReLU())

    def forward(self, graph_tensors):
        f_nodes, f_edges, node2edge, edge2node, b2revb = graph_tensors
        # a2a = edge2node[node2edge]  # num_atoms x max_num_bonds
        # f_edges = f_edges[:, -self.edge_dim:]
        f_edges = self.w_edge(f_edges)
        f_nodes = f_nodes.view(-1)
        node_feats = self.edit_embedding(f_nodes)  # node_init features
        n_input = f_edges
        n_input[0] = 0
        # return n_input
        message = n_input
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding
        for depth in range(self.depth - 1):
            # # num_atoms x max_num_bonds x hidden  message -> hidden
            # # num_atoms x max_num_bonds x hidden
            # nei_a_message = index_select_ND(message, a2a)
            # # num_atoms x max_num_bonds x bond_fdim
            # nei_f_bonds = index_select_ND(f_edges, node2edge)
            # # num_atoms x max_num_bonds x hidden + bond_fdim
            # nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
            # # num_atoms x hidden + bond_fdim
            # message = nei_message.sum(dim=1)
            # message = self.w_h(message)  # num_bonds x hidden\
            nei_a_message = index_select_ND(message, node2edge)
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden sum(h_ki)
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[edge2node] - rev_message  # num_bonds x hidden

            message = self.gru(n_input, message)  # num_bonds x hidden_size U function
            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden
        # nei_a_message = index_select_ND(message, a2a)
        nei_a_message = index_select_ND(message, node2edge)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([node_feats, a_message], dim=1)
        atom_hiddens = self.read_out(a_input)  # num_atoms x hidden
        return atom_hiddens


class RandomInit(nn.Module):
    def __init__(self, hidden_size, depth, node_f_dim, edge_f_dim, dropout, vocab_size, edge_init_dim):
        super(RandomInit, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.node_dim = node_f_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.edit_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=node_f_dim)

    def forward(self, graph_tensors):
        f_nodes, f_edges, node2edge, edge2node, b2revb = graph_tensors
        a_message = self.edit_embedding(f_nodes)
        return a_message.view(a_message.size(0), -1)
