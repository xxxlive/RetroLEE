from typing import Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_data_rdkit_only import apply_edit_to_mol
from rdkit import Chem
from utils.collate_fn import get_batch_graphs
from utils.datasets import generate_sample_idx
from utils.rxn_graphs import MolGraph, Vocab

from models.encoder import Global_Attention, MPNEncoder
from models.model_utils import (creat_edits_feats, index_select_ND,
                                unbatch_feats, get_tmp_edit_tensor)

import numpy as np


class RetroLEE(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 tot_vocab: Vocab,
                 transit_graph=None,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(RetroLEE, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.device = device
        self.transit_graph = transit_graph
        self.tot_vocab = tot_vocab
        self.bond_fdim = config['n_bond_feat']
        self.contrastive_loss = config['contrastive_loss']
        self.temperature = config['t']
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        self.atom_hidden_size = config['mpn_size']
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                  bond_fdim=config['n_bond_feat'],
                                  hidden_size=self.atom_hidden_size,
                                  depth=config['depth'],
                                  dropout=config['dropout_mpn'],
                                  atom_message=config['atom_message'],
                                  model_type=config['model_type'])
        self.editEncoder = MPNEncoder(atom_fdim=256,
                                      bond_fdim=128,
                                      hidden_size=config['mpn_size'],
                                      depth=2,
                                      dropout=config['dropout_mpn'],
                                      atom_message=config['atom_message'],
                                      model_type='random', vocab_size=len(self.tot_vocab), edge_init_dim=2)
        self.W_vv = nn.Linear(self.atom_hidden_size,
                              self.atom_hidden_size, bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(self.atom_hidden_size,
                              self.atom_hidden_size, bias=False)
        nn.init.eye_(self.W_vc.weight)

        self.W_ve = nn.Sequential(nn.ReLU(), nn.Linear(config['mpn_size'],
                                                       config['mpn_size'], bias=False))
        self.W_ea = nn.Sequential(
            nn.Linear(self.atom_hidden_size + config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_hidden_size),
            nn.ReLU()
        )
        # self.W_lg_vc = nn.Linear(config['mpn_size'],
        #                          config['mpn_size'], bias=False)
        # nn.init.eye_(self.W_ve.linear.weight)
        if config['use_attn']:
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])
        self.atom_linear = nn.Sequential(
            nn.Linear(self.atom_hidden_size, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))
        self.bond_linear = nn.Sequential(
            nn.Linear(self.atom_hidden_size * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))
        self.graph_linear = nn.Sequential(
            nn.Linear(self.atom_hidden_size, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))
        self.bond_feature_ex = nn.Sequential(
            nn.Linear(self.atom_hidden_size * 2 + self.bond_fdim, config['mpn_size'] * 2),
            nn.ReLU(),
            nn.Dropout(config['dropout_mlp']),
            nn.Linear(config['mpn_size'] * 2, self.atom_hidden_size * 2).to(self.device),
        )
        self.edit_conf = nn.Sequential(nn.Linear(self.atom_hidden_size + config['mpn_size'], 1))
        self.W_ve_out = nn.Sequential(nn.ReLU(), nn.Linear(config['mpn_size'],
                                                           config['mpn_size'], bias=False))
        # self.edit_conf = nn.Sequential(nn.Linear(config['mpn_size'] * 2, config['mpn_size']), nn.ReLU(),
        #                                nn.Linear(config['mpn_size'], 1))
        # self.atom_selector = FocusSelector(in_feature=config['mpn_size'], drop_out=config['dropout_mlp'])
        # self.bond_selector = FocusSelector(in_feature=config['mpn_size'], drop_out=config['dropout_mlp'])

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def cal_transit_embedding(self, edit_data):
        if self.transit_graph is not None:
            f_nodes, f_edges, node2edge, edge2node, b2revb = self.transit_graph
        else:
            f_nodes = torch.arange(0, len(self.tot_vocab) + 1, device=self.device, dtype=torch.long)
            f_nodes, f_edges, node2edge, edge2node, b2revb = f_nodes.unsqueeze(1), None, None, None, None
        f_nodes = f_nodes - 1
        f_nodes[0] = 0
        edit_embedding = self.editEncoder(
            (f_nodes, f_edges, node2edge, edge2node, b2revb), mask=None)
        res_edit = index_select_ND(edit_embedding[1:], edit_data)
        return res_edit

    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],
                            prod_scopes: Tuple[List], prev_atom_hiddens: torch.Tensor = None,
                            prev_atom_scope: Tuple[List] = None,
                            return_output=False, edit_data=None) -> Tuple[
        torch.Tensor]:
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = prod_tensors
        # last step edit info
        edit_data_a, last_edit_mask = edit_data
        edit_data_a = self.to_device(edit_data_a)
        last_edit_mask = self.to_device(last_edit_mask)
        assert edit_data_a.shape[0] == last_edit_mask.shape[0]
        # edit_embedding = self.edit_embedding(edit_data).view(f_atoms.shape[0], -1)
        # edit_embedding[0, :] = 0
        edit_embedding = self.cal_transit_embedding(edit_data_a)
        edit_embedding = edit_embedding.view(f_atoms.shape[0], -1)
        edit_embedding[0, :] = 0
        atom_mask_embedding = self.cal_transit_embedding(last_edit_mask)
        atom_mask_embedding = atom_mask_embedding.view(last_edit_mask.size(0), -1)
        atom_mask_embedding[0, :] = 0

        prod_tensors = self.to_device(prod_tensors)
        atom_scope, bond_scope = prod_scopes

        if prev_atom_hiddens is None:
            n_atoms = prod_tensors[0].size(0)
            prev_atom_hiddens = torch.zeros(
                n_atoms, self.atom_hidden_size, device=self.device)
        a_feats = self.encoder(prod_tensors, mask=None, edit_embedding=edit_embedding)
        # global attention
        if self.config['use_attn']:
            feats, mask = creat_edits_feats(a_feats, atom_scope)
            attention_score, feats = self.attn(feats, mask)
            a_feats = unbatch_feats(feats, atom_scope)
        # if atom numbers not equal add new nodes
        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:
            n_atoms = a_feats.shape[0]
            new_ha = torch.zeros(
                n_atoms, self.atom_hidden_size, device=self.device)
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]
            prev_atom_hiddens = new_ha

        assert a_feats.shape == prev_atom_hiddens.shape

        ori_graph_vecs = torch.stack(
            [a_feats[st: st + le].mean(dim=0) for st, le in atom_scope])
        # pooling atom edit embedding

        # a_mask = (a2a > 0).sum(axis=1).unsqueeze(-1)
        # a_mask[a_mask == 0] = 1
        # atom_edit_embedding = index_select_ND(edit_embedding, a2a).sum(dim=1) / a_mask
        # atom_edit_embedding = index_select_ND(edit_embedding, a2a).sum(dim=1)
        # fusion the last step feature
        atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))
        # 让原子自己输出当前的关注度
        # atom_feats + edit_embedding
        a2a = b2a[a2b].to(self.device)
        self_loop = torch.arange(0, edit_embedding.shape[0]).to(self.device)
        a2a = torch.cat([self_loop.unsqueeze(1), a2a], dim=1)
        edit_importance = torch.cat([edit_embedding, atom_feats], dim=1)
        edit_importance = self.edit_conf(edit_importance)
        edit_importance = index_select_ND(edit_importance, a2a)
        atom_edit_embedding = index_select_ND(edit_embedding, a2a)
        atom_edit_embedding[0] = 0
        atom_edit_embedding = (edit_importance * atom_edit_embedding).sum(dim=1)
        atom_feats = self.W_ea(
            torch.cat([atom_feats, self.W_ve(atom_edit_embedding), self.W_ve_out(atom_mask_embedding)], dim=-1))

        # atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))
        prev_atom_hiddens = atom_feats.clone()
        prev_atom_scope = atom_scope

        bond_feats = self.get_bond_feature(prod_tensors, atom_feats)
        graph_vecs = torch.stack(
            [atom_feats[st: st + le].mean(dim=0) for st, le in atom_scope])
        node_feats = atom_feats.clone()
        # select atoms
        # atom_selected = self.atom_selector(node_feats, atom_scope)
        # atom_l, bond_l, graph_l = self.build_edit_mask_embedding(last_edit_mask, prod_scopes, bond_feats)
        atom_outs = self.atom_linear(torch.cat([node_feats], dim=1))
        bond_outs = self.bond_linear(torch.cat([bond_feats], dim=1))
        graph_outs = self.graph_linear(torch.cat([graph_vecs], dim=1))

        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]

        if return_output:
            # 支持单个product
            return edit_scores, prev_atom_hiddens, prev_atom_scope, (node_feats, bond_feats, ori_graph_vecs)
        return edit_scores, prev_atom_hiddens, prev_atom_scope

    def build_edit_mask_embedding(self, atom_mask, scope, bond_feats):
        atom_scope, bond_scope = scope
        bond_mask = torch.zeros(bond_feats.size(0))
        graph_mask = torch.zeros(len(atom_scope))
        for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope))):
            edit_idx = atom_mask[st_a]
            bond_mask[st_b:st_b + le_b] = edit_idx
            graph_mask[idx] = edit_idx
        bond_mask = self.to_device(bond_mask).long()
        graph_mask = self.to_device(graph_mask).long()
        atom_mask_embedding = self.cal_transit_embedding(atom_mask)
        atom_mask_embedding = atom_mask_embedding.view(atom_mask.size(0), -1)
        atom_mask_embedding[0, :] = 0
        bond_mask_embedding = self.cal_transit_embedding(bond_mask)
        bond_mask_embedding = bond_mask_embedding.view(bond_feats.size(0), -1)
        bond_mask_embedding[0, :] = 0
        graph_mask_embedding = self.cal_transit_embedding(graph_mask)
        return atom_mask_embedding, bond_mask_embedding, graph_mask_embedding

    def get_bond_feature(self, prod_tensors, atom_feature):
        f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a = prod_tensors
        atom1_st = index_select_ND(atom_feature, undirected_b2a[:, 0])  # hi
        atom2_ed = index_select_ND(atom_feature, undirected_b2a[:, 1])  # hj
        sum_atom_vecs = atom1_st + atom2_ed
        diff_atom_vecs = torch.abs(atom1_st - atom2_ed)
        bond_selected = torch.zeros(atom1_st.shape[0])
        # 效率太低要改掉
        for i in range(1, undirected_b2a.shape[0]):
            bond_idx = a2b[undirected_b2a[i, 0]]
            bond_selected[i] = torch.where(b2a[bond_idx] == undirected_b2a[i, 1], bond_idx, 0).sum()
        bond_feature = f_bonds[:, -self.bond_fdim:]
        bond_feature = bond_feature[bond_selected.long(), :]
        bond_feature = self.bond_feature_ex(torch.cat((bond_feature, sum_atom_vecs, diff_atom_vecs), dim=1))
        return bond_feature

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]], seq_mask, edit_data) -> Tuple[torch.Tensor]:
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence

        """
        max_seq_len = len(prod_seq_inputs)
        assert len(prod_seq_inputs[0]) == 2
        prev_atom_hiddens = None
        prev_atom_scope = None
        seq_edit_scores = []
        graph_vectors = [[] for _ in range(len(prod_seq_inputs[0][1][0]))]
        for idx in range(max_seq_len):
            prod_tensors, prod_scopes = prod_seq_inputs[idx]
            cur_edit_data = edit_data[idx]
            edit_scores, prev_atom_hiddens, prev_atom_scope, all_hiddens = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope, return_output=True,
                edit_data=cur_edit_data)
            graph_vectors = self.store_graph_tensors(graph_vectors, all_hiddens, seq_mask[idx, :])
            seq_edit_scores.append(edit_scores)
        lg_loss = self.cal_graph_sim(graph_vectors, type=self.contrastive_loss)
        # lg_loss = torch.tensor(0)
        return seq_edit_scores, lg_loss

    # def cal_clip_sim(self, prod_graph, syn_graph):
    #     prod_graph = F.normalize(prod_graph, dim=-1)
    #     # prod_graph = prod_graph / prod_graph.norm(dim=1, keepdim=True)
    #     # syn_graph = syn_graph / syn_graph.norm(dim=1, keepdim=True)
    #     syn_graph = F.normalize(syn_graph, dim=-1)
    #     logit1 = self.logit_scale.exp() * prod_graph @ syn_graph.T
    #     logit2 = self.logit_scale.exp() * syn_graph @ prod_graph.T
    #     label = torch.tensor(np.arange(0, logit1.shape[0]), device=logit1.device)
    #     loss = (F.cross_entropy(logit1, label) + F.cross_entropy(logit2, label)) / 2
    #     return loss / logit1.shape[0]
    def cal_clip_sim(self, prod_graph, syn_graph):
        # prod_graph = F.normalize(prod_graph, dim=-1)
        prod_graph = prod_graph / prod_graph.norm(dim=1, keepdim=True)
        syn_graph = syn_graph / syn_graph.norm(dim=1, keepdim=True)
        # syn_graph = F.normalize(syn_graph, dim=-1)
        logit1 = prod_graph @ syn_graph.T
        logit2 = prod_graph @ prod_graph.T
        # hard negative mining
        positive_sample = torch.diag(logit1)
        # remove diag
        logit2 = logit2.fill_diagonal_(-100000000)
        # sorted_loss_indices = torch.argsort(logit1, descending=True)
        max_values, max_indices = torch.max(logit2, dim=1)
        rows = torch.arange(logit2.size(0))
        negative_sample = logit2[rows, max_indices]
        samples = torch.stack([positive_sample, negative_sample], dim=1)
        # sorted_loss_indices = sorted_loss_indices[:1]
        label = torch.zeros((logit1.size(0)), dtype=torch.long).to(samples.device)
        # label = torch.tensor(np.arange(0, logit1.shape[0]), device=logit1.device)
        # loss = (F.cross_entropy(logit1, label) + F.cross_entropy(logit2, label)) / 2
        # return loss / logit1.shape[0]
        loss_pos = F.cross_entropy(samples, label)
        # loss_neg = F.binary_cross_entropy(negative_sample, torch.zeros(positive_sample.shape))
        return 0.2 * loss_pos

    def _get_correlated_mask(self, batch_size, device):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(device)

    def cal_nxtloss(self, prod_graph, syn_graph):
        batch_size = prod_graph.shape[0]
        prod_graph = prod_graph / prod_graph.norm(dim=1, keepdim=True)
        syn_graph = syn_graph / syn_graph.norm(dim=1, keepdim=True)
        representations = torch.cat([prod_graph, syn_graph], dim=0)
        similarity_matrix = representations @ representations.T
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size, prod_graph.device).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(prod_graph.device).long()
        loss = F.cross_entropy(logits, labels)

        return 0.2 * loss / (2 * batch_size)

    def cal_mse_contrasitive(self, prod_graph, syn_graph):
        bsz = prod_graph.size(0)
        pos_l1 = F.mse_loss(prod_graph, syn_graph)
        # 从syngraph中采样 随机采样
        neg_idx = generate_sample_idx(bsz)
        neg_sample = prod_graph[neg_idx]
        neg_l2 = F.mse_loss(prod_graph, neg_sample)
        loss = 0.2 * (pos_l1 / (neg_l2 + pos_l1))
        return loss

    def cal_graph_sim(self, graph_vectors, type='ss'):
        prod_graph = torch.stack([graph_vectors[i][0] for i in range(len(graph_vectors))], dim=0)
        syn_graph = torch.stack([graph_vectors[i][1] for i in range(len(graph_vectors))], dim=0)
        if type == 'clip':
            loss = self.cal_clip_sim(prod_graph, syn_graph)
        elif type == 'mse':
            loss = self.cal_mse_contrasitive(prod_graph, syn_graph)
        elif type == 'nxt':
            loss = self.cal_nxtloss(prod_graph, syn_graph)
        return loss

    def store_graph_tensors(self, graph_vectors, all_hiddens, seq_mask):
        f_out, b_out, g_out = all_hiddens
        for i in range(g_out.shape[0]):
            if seq_mask[i] == 1:
                graph_vectors[i].append(g_out[i, :])
        return graph_vectors

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 9):
        """Make predictions for given product smiles string.
        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        done = False
        steps = 0
        edits = []
        edits_atom = []
        prev_atom_hiddens = None
        prev_atom_scope = None

        products = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(products)
        prod_graph = MolGraph(mol=Chem.Mol(products),
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        prod_tensors, prod_scopes = get_batch_graphs(
            [prod_graph], use_rxn_class=use_rxn_class)
        # lg_interface_info = (list(lg_info[:]) + [None])

        cur_edit_tensors = get_tmp_edit_tensor(prod_tensors, None, None, prod_graph, self.tot_vocab)
        while not done and steps <= max_steps:
            if prod_tensors[-1].size() == (1, 0):
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            edit_logits, prev_atom_hiddens, prev_atom_scope, feature_tensors = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope, return_output=True,
                edit_data=cur_edit_tensors)
            idx = torch.argmax(edit_logits[0])
            # 测试代码
            # idx = torch.tensor([79])[0]
            val = edit_logits[0][idx]

            max_bond_idx = products.GetNumBonds() * self.bond_outdim

            if idx.item() == len(edit_logits[0]) - 1:
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            elif idx.item() < max_bond_idx:
                bond_logits = edit_logits[0][:products.GetNumBonds(
                ) * self.bond_outdim]
                bond_logits = bond_logits.reshape(
                    products.GetNumBonds(), self.bond_outdim)
                idx_tensor = torch.where(bond_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = products.GetBondWithIdx(
                    bond_idx).GetBeginAtom().GetAtomMapNum()
                a2 = products.GetBondWithIdx(
                    bond_idx).GetEndAtom().GetAtomMapNum()

                a1, a2 = sorted([a1, a2])
                edit_atom = [a1, a2]
                edit = self.bond_vocab.get_elem(edit_idx)

            else:
                atom_logits = edit_logits[0][max_bond_idx:-1]

                assert len(atom_logits) == products.GetNumAtoms() * \
                       self.atom_outdim
                atom_logits = atom_logits.reshape(
                    products.GetNumAtoms(), self.atom_outdim)
                idx_tensor = torch.where(atom_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()
                edit_atom = a1
                edit = self.atom_vocab.get_elem(edit_idx)

            try:
                products = apply_edit_to_mol(mol=Chem.Mol(
                    products), edit=edit, edit_atom=edit_atom)
                prod_graph = MolGraph(mol=Chem.Mol(
                    products), rxn_class=rxn_class, use_rxn_class=use_rxn_class)
                prod_tensors, prod_scopes = get_batch_graphs(
                    [prod_graph], use_rxn_class=use_rxn_class)
                edits.append(edit)
                edits_atom.append(edit_atom)
                steps += 1
                cur_edit_tensors = get_tmp_edit_tensor(prod_tensors, edit, edit_atom, prod_graph, self.tot_vocab)
                # print(cur_edit_tensors.shape)
            except:
                steps += 1
                continue

        return edits, edits_atom

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['transit_graph'] = self.transit_graph
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab
        saveables['tot_vocab'] = self.tot_vocab

        return saveables
