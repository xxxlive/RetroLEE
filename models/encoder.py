from typing import Tuple
import math
import torch
import torch.nn as nn

from models.gcn_encoder import GCN, GCNLight
from models.gin_encoder import GINE
from models.gru_encoder import GRUMPNN, GRULight, RandomInit
from models.model_utils import index_select_ND


class MPNEncoder(nn.Module):
    """Class: 'MPNEncoder' is a message passing neural network for encoding molecules."""

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 depth: int, dropout: float = 0.15, atom_message: bool = False, model_type='gru', **kwargs):
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
        atom_message: 'D-MPNN' or 'MPNN', centers messages on bonds or atoms.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        # bond for self-loop
        if model_type == 'gin':
            self.bond_fdim += 1
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.atom_message = atom_message
        self.model_type = model_type
        # Input
        input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        # self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)
        #
        # # Update message
        # if self.atom_message:
        #     self.w_h = nn.Linear(
        #         self.bond_fdim + self.hidden_size, self.hidden_size)
        if model_type == 'gru':
            # self.msg_passing = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.msg_passing = GRUMPNN(hidden_size=hidden_size, depth=depth, atom_message=atom_message,
                                       atom_fdim=atom_fdim, bond_fdim=self.bond_fdim, dropout=dropout)
        elif model_type == 'gcn':
            self.msg_passing = GCN(hidden_size=hidden_size, depth=depth, atom_message=atom_message,
                                   atom_fdim=atom_fdim, bond_fdim=self.bond_fdim, dropout=dropout)
        elif model_type == 'gin':
            self.msg_passing = GINE(hidden_size=hidden_size, depth=depth, atom_message=atom_message,
                                    atom_fdim=atom_fdim,
                                    bond_fdim=self.bond_fdim, dropout=dropout)
        elif model_type == 'gru_edit':
            self.msg_passing = GRULight(hidden_size=hidden_size, depth=depth, node_f_dim=atom_fdim,
                                        edge_f_dim=bond_fdim, dropout=dropout,
                                        vocab_size=kwargs['vocab_size'], edge_init_dim=kwargs['edge_init_dim'])
        elif model_type == 'gcn_edit':
            self.msg_passing = GCNLight(hidden_size=hidden_size, depth=depth, node_f_dim=atom_fdim,
                                        edge_f_dim=bond_fdim, dropout=dropout,
                                        vocab_size=kwargs['vocab_size'], edge_init_dim=kwargs['edge_init_dim'])
        elif model_type == 'random':
            self.msg_passing = RandomInit(hidden_size=hidden_size, depth=depth, node_f_dim=atom_fdim,
                                          edge_f_dim=bond_fdim, dropout=dropout,
                                          vocab_size=kwargs['vocab_size'], edge_init_dim=kwargs['edge_init_dim'])
        # self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        #
        # # Dropout
        # self.dropout_layer = nn.Dropout(p=self.dropout)
        # Output
        # self.W_o = nn.Sequential(
        #     nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.ReLU())

    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor, edit_embedding=None) -> torch.FloatTensor:
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        atom_hiddens = self.msg_passing(graph_tensors)
        # a_input = torch.cat([f_atoms, a_message], dim=1)
        # atom_hiddens = self.W_o(a_input)  # num_atoms x hidden

        if mask is None:
            mask = torch.ones(atom_hiddens.size(0), 1, device=atom_hiddens.device)
            mask[0, 0] = 0  # first node is padding

        return atom_hiddens * mask
        # Input
        # if self.atom_message:
        #     a2a = b2a[a2b]  # num_atoms x max_num_bonds
        #     f_bonds = f_bonds[:, -self.bond_fdim:]
        #     input = self.w_i(f_atoms)  # num_atoms x hidden
        # else:
        #     input = self.w_i(f_bonds)  # num_bonds x hidden
        #
        # # Message passing
        # # message = torch.zeros(input.size(0), self.hidden_size, device=input.device)
        # message = input
        # message_mask = torch.ones(message.size(0), 1, device=message.device)
        # message_mask[0, 0] = 0  # first message is padding
        # multi_layer_message = []
        # # mpnn层数
        # for depth in range(self.depth - 1):
        #     if self.atom_message:
        #         # num_atoms x max_num_bonds x hidden
        #         nei_a_message = index_select_ND(message, a2a)
        #         # num_atoms x max_num_bonds x bond_fdim
        #         nei_f_bonds = index_select_ND(f_bonds, a2b)
        #         # num_atoms x max_num_bonds x hidden + bond_fdim
        #         nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
        #         # num_atoms x hidden + bond_fdim
        #         message = nei_message.sum(dim=1)
        #         message = self.w_h(message)  # num_bonds x hidden
        #     else:
        #         # num_atoms x max_num_bonds x hidden  message -> hidden
        #         nei_a_message = index_select_ND(message, a2b)
        #         a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden sum(h_ki)
        #         rev_message = message[b2revb]  # num_bonds x hidden
        #         message = a_message[b2a] - rev_message  # num_bonds x hidden
        #         # a_message[b2a] = sum(ki) - hij的对称边
        #     message = self.msg_passing(input, message)  # num_bonds x hidden_size U function
        #     message = message * message_mask
        #     message = self.dropout_layer(message)  # num_bonds x hidden
        #     if self.model_type == 'gcn':
        #         multi_layer_message.append(message)

        # if self.atom_message:
        #     # num_atoms x max_num_bonds x hidden
        #     nei_a_message = index_select_ND(message, a2a)
        # else:
        #     # num_atoms x max_num_bonds x hidden
        #     nei_a_message = index_select_ND(message, a2b)
        # a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # num_atoms x (atom_fdim + hidden)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores[~mask.bool()] = float(-9e15)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class Global_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):
        super(Global_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x
