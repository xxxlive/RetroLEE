from typing import Tuple
import math
import torch
import torch.nn as nn

from models.model_utils import index_select_ND


class GCN(nn.Module):
    def __init__(self, hidden_size, depth, atom_message, atom_fdim, bond_fdim, dropout):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.atom_message = atom_message
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.depth = depth
        self.dropout = dropout
        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        self.W_g = nn.Sequential(
            nn.Linear(input_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_mess = nn.Sequential(
            nn.Linear(depth * hidden_size, int(depth / 2) * hidden_size),
            nn.ReLU(),
            nn.Linear(int(depth / 2) * hidden_size, hidden_size))
        self.W_o = nn.Sequential(
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU())

    def gcn(self, fmess, nei_message):
        g_input = torch.cat([fmess, nei_message], dim=1)
        messages = self.W_g(g_input)
        return messages

    def forward(self, graph_tensors):
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = graph_tensors
        message = torch.zeros(f_bonds.size(0), self.hidden_size, device=f_bonds.device)
        # message = f_bonds
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding
        multi_layer_message = []
        # mpnn层数
        for depth in range(self.depth):
            # num_atoms x max_num_bonds x hidden  message -> hidden
            nei_a_message = index_select_ND(message, a2b)
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden sum(h_ki)
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden
            # a_message[b2a] = sum(ki) - hij的对称边
            message = self.gcn(f_bonds, message)  # num_bonds x hidden_size U function
            message = message * message_mask
            message = self.dropout_layer(message)  # num_bonds x hidden
            multi_layer_message.append(message)
        tmp_messages = torch.cat(multi_layer_message, dim=1)
        tmp_messages = self.output_mess(tmp_messages)
        nei_a_message = index_select_ND(tmp_messages, a2b)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden
        return atom_hiddens


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edge_feature, bias=True):
        super(GraphConvolution, self).__init__()
        self.node_weight = nn.Linear(in_features, out_features)

        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.normal_(self.weight)
        # nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, graph_tensors, fedges, a2a):
        f_nodes, f_bonds, node2edge, edge2node, b2revb = graph_tensors

        support = self.node_weight(input_features)
        # num_atoms x max_num_bonds x hidden  message -> hidden
        # num_atoms x max_num_bonds x hidden
        nei_a_message = index_select_ND(support, a2a)
        # num_atoms x max_num_bonds x bond_fdim
        nei_f_bonds = index_select_ND(fedges, node2edge)
        # num_atoms x max_num_bonds x hidden + bond_fdim
        nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
        nei_message = nei_a_message * nei_f_bonds
        # num_atoms x hidden + bond_fdim
        message = nei_message.sum(dim=1)
        # message = self.weight(message)  # num_bonds x hidden

        output = torch.tanh(message)
        # print('output:',output)
        if self.use_bias:
            # print(self.bias)
            return output + self.bias
        else:
            return output


class GCNLight(nn.Module):
    def __init__(self, hidden_size, depth, node_f_dim, edge_f_dim, dropout, vocab_size, edge_init_dim):
        super(GCNLight, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.node_dim = node_f_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.edit_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=node_f_dim)
        self.edge_dim = edge_init_dim
        self.w_h = nn.Linear(edge_f_dim + node_f_dim, node_f_dim)
        self.gcn1 = GraphConvolution(node_f_dim, node_f_dim * 2, edge_feature=edge_f_dim)
        self.gcn2 = GraphConvolution(node_f_dim * 2, node_f_dim, edge_feature=edge_f_dim)

    def forward(self, graph_tensors):
        f_nodes, f_edges, node2edge, edge2node, b2revb = graph_tensors
        a2a = edge2node[node2edge]  # num_atoms x max_num_bonds
        f_edges = f_edges[:, -self.edge_dim:]
        # f_edges = self.w_edge(f_edges)
        f_nodes = f_nodes.view(-1)
        n_input = self.edit_embedding(f_nodes)
        n_input[0] = 0
        # return n_input
        return n_input
        node_embedding = n_input
        message = node_embedding
        message_mask = torch.ones(message.size(0), 1, device=message.device)
        message_mask[0, 0] = 0  # first message is padding
        message = self.gcn1(n_input, graph_tensors, f_edges, a2a)
        message = self.gcn2(message, graph_tensors, f_edges, a2a)
        afeats = message
        # print('hello')
        # first layer
        # for depth in range(self.depth - 1):
        #     # num_atoms x max_num_bonds x hidden  message -> hidden
        #     # num_atoms x max_num_bonds x hidden
        #     nei_a_message = index_select_ND(message, a2a)
        #     # num_atoms x max_num_bonds x bond_fdim
        #     nei_f_bonds = index_select_ND(f_edges, node2edge)
        #     # num_atoms x max_num_bonds x hidden + bond_fdim
        #     nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
        #     # num_atoms x hidden + bond_fdim
        #     message = nei_message.sum(dim=1)
        #     message = self.w_h(message)  # num_bonds x hidden
        #
        #     message = self.gru(n_input, message)  # num_bonds x hidden_size U function
        #     message = message * message_mask
        #     message = self.dropout_layer(message)  # num_bonds x hidden
        # nei_a_message = index_select_ND(message, a2a)
        # a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # afeats = torch.cat([n_input, a_message], dim=1)
        # afeats = self.read_out(afeats)
        # return 0
        return afeats * message_mask
