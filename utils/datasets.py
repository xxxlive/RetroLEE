import copy
import os
from random import random
from typing import List, Optional, Tuple, Any

import joblib
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from prepare_data_rdkit_only import apply_edit_to_mol
from utils.chem import compare_mol
from utils.collate_fn import prepare_edit_labels, get_batch_graphs, get_batch_graphs_lgs, build_edit_tensors
from utils.generate_edits import ReactionData
import rdkit.Chem as Chem

from utils.reaction_actions import Termination
from utils.rxn_graphs import Vocab, MolGraph, RxnGraph


def generate_sample_idx(n):
    arr = []
    for i in range(n):
        while True:
            num = np.random.randint(0, n)
            if num != i:
                arr.append(num)
                break
    return arr


class RetroEditDataset(Dataset):
    def __init__(self, data_dir: str, **kwargs):
        self.data_dir = data_dir
        self.data_files = [
            os.path.join(self.data_dir, file)
            for file in os.listdir(self.data_dir)
            if "batch-" in file
        ]
        # only add in test
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Retrieves a particular batch of tensors.

        Parameters
        ----------
        idx: int,
            Batch index
        """
        batch_tensors = torch.load(self.data_files[idx], map_location='cpu')

        return batch_tensors

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        return len(self.data_files)

    def collate(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:
        """Processes the batch of tensors to yield corresponding inputs."""
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        graph_seq_tensors, edit_seq_labels, seq_mask = attributes
        return graph_seq_tensors, edit_seq_labels, seq_mask

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate)


class RetroEvalDataset(Dataset):
    def __init__(self, data_dir: str, data_file: str, use_rxn_class: bool = False):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file)
        self.use_rxn_class = use_rxn_class
        self.dataset = joblib.load(self.data_file)

    def __getitem__(self, idx: int) -> ReactionData:
        """Retrieves the corresponding ReactionData

        Parameters
        ----------
        idx: int,
        Index of particular element
        """
        return self.dataset[idx]

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        # return 10
        return len(self.dataset)

    def collate(self, attributes: List[ReactionData]) -> Tuple[str, List[Tuple], List[List], Optional[List[int]]]:
        """Processes the batch of tensors to yield corresponding inputs."""
        rxns_batch = attributes
        prod_smi = [rxn_data.rxn_smi.split(">>")[-1]
                    for rxn_data in rxns_batch]
        edits = [rxn_data.edits for rxn_data in rxns_batch]
        edits_atom = [rxn_data.edits_atom for rxn_data in rxns_batch]

        if self.use_rxn_class:
            rxn_classes = [rxn_data.rxn_class for rxn_data in rxns_batch]
            return prod_smi, edits, edits_atom, rxn_classes
        else:
            return prod_smi, edits, edits_atom, None

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> DataLoader:
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate)


class RetroRandomEditDataset(Dataset):
    def __init__(self, data_dir: str, max_steps: int, use_rxn, lg_random_aug=False, **kwargs):
        self.data_dir = data_dir
        bond_vocab_file = os.path.join(self.data_dir, 'bond_vocab.txt')
        atom_only_vocab_file = os.path.join(self.data_dir, 'atom_vocab.txt')
        lg_only_vocab_file = os.path.join(self.data_dir, 'lg_vocab.txt')
        atom_vocab_file = os.path.join(self.data_dir, 'atom_lg_vocab.txt')
        self.bond_vocab = Vocab(joblib.load(bond_vocab_file))
        self.atom_vocab = Vocab(joblib.load(atom_vocab_file))
        self.tot_vocab = Vocab(
            joblib.load(atom_vocab_file) + joblib.load(bond_vocab_file) + ['Terminate', 'Initialize', 'Null'])
        self.atom_only_vocab = Vocab(joblib.load(atom_only_vocab_file))
        self.lg_only_vocab = Vocab(joblib.load(lg_only_vocab_file))
        # self.tot_vocab = self.bond_vocab + self.atom_vocab
        self.use_rxn_class = use_rxn
        if use_rxn:
            self.batch_dir = os.path.join(self.data_dir, 'with_rxn_class')
        else:
            self.batch_dir = os.path.join(self.data_dir, 'without_rxn_class')
        self.rxn_arr = [file for file in os.listdir(self.batch_dir)
                        if "batch_" in file]
        self.max_steps = max_steps
        # only add in test
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Retrieves a particular batch of tensors.

        Parameters
        ----------
        idx: int,
            Batch index
        """
        item = joblib.load(os.path.join(self.batch_dir, f"tot_batch_data_{idx}.pt"))
        return item

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        # return 1000
        return len(self.rxn_arr)

    def collate(self, rxn_list) -> Tuple[Any, Any, Any, Any]:
        """Processes the batch of tensors to yield corresponding inputs."""
        batch_tensors = self.process_batch_data(rxn_list)
        graph_seq_tensors, edit_seq_labels, seq_mask, edit_seq_tensors = batch_tensors
        return graph_seq_tensors, edit_seq_labels, seq_mask, edit_seq_tensors

    def random_lg(self, batch_graphs):
        for i in range(len(batch_graphs)):
            cur_seq = batch_graphs[i]
            # 统计有多少个lg
            cnt = 0
            for j in range(len(cur_seq)):
                if cur_seq[j].edit_to_apply[0] == 'Attaching LG':
                    cnt += 1
            if cnt < 2:
                continue
            else:
                batch_graphs[i] = self.reconstruct_molseq(cur_seq)
        return batch_graphs

    def process_batch_data(self, batch_graphs):
        lengths = torch.tensor([len(graph_seq)
                                for graph_seq in batch_graphs], dtype=torch.long)
        max_length = max([len(graph_seq) for graph_seq in batch_graphs])
        graph_seq_tensors = []
        edit_seq_labels = []
        edit_seq_tensros = []
        seq_mask = []
        last_edits, last_edit_atoms = None, None
        for idx in range(max_length):
            graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length - 1)]).get_components(
                attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                for i, length in enumerate(lengths)]
            mask = (idx < lengths).long()
            prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))
            assert all([isinstance(graph, MolGraph) for graph in prod_graphs])
            # 获取标签序列
            edit_labels, lg_edits = prepare_edit_labels(
                prod_graphs, edits, edit_atoms, self.bond_vocab, self.atom_vocab)
            # 将图拼接为一个大图
            current_graph_tensors = get_batch_graphs(prod_graphs, use_rxn_class=self.use_rxn_class)
            current_edit_tensors, current_edit_mask = build_edit_tensors(current_graph_tensors,
                                                                         (last_edits, last_edit_atoms),
                                                                         is_initial=(idx == 0),
                                                                         prod_graphs=prod_graphs,
                                                                         vocab=self.tot_vocab)
            last_edits, last_edit_atoms = edits, edit_atoms
            # current_graph_tensors, current_lg_tensors = get_batch_graphs_lgs(
            #     prod_graphs, use_rxn_class=self.use_rxn_class, lg_edits=lg_edits, add_cls=False)
            # 把相同的lg进行mask操作
            graph_seq_tensors.append(current_graph_tensors)
            edit_seq_tensros.append((current_edit_tensors, current_edit_mask))
            edit_seq_labels.append(edit_labels)
            seq_mask.append(mask)

        seq_mask = torch.stack(seq_mask).long()
        assert seq_mask.shape[0] == max_length
        assert seq_mask.shape[1] == len(batch_graphs)

        return graph_seq_tensors, edit_seq_labels, seq_mask, edit_seq_tensros

    def generate_lg_mask(self, lg_edits, lg_idx):
        lg_cand = [lg_edits[i] for i in lg_idx]
        length = len(lg_idx)
        mask = np.zeros((length, length))
        for i in range(len(lg_cand)):
            for j in range(i, len(lg_cand)):
                if lg_cand[i] == lg_cand[j]:
                    mask[i][j] = mask[j][i] = 1
        mask = torch.FloatTensor(mask)
        return mask

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate)
