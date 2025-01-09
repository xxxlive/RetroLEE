import argparse
import os
import random
import sys

import joblib
from datetime import datetime as dt

from models.retrolee import RetroLEE
import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger, Chem
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from models.model_utils import CSVLogger, get_seq_edit_accuracy
from utils.collate_fn import get_batch_graphs
from utils.datasets import RetroEditDataset, RetroEvalDataset, RetroRandomEditDataset
from utils.mol_features import ATOM_FDIM, BOND_FDIM
from utils.reaction_actions import judge
from utils.rxn_graphs import Vocab, MolGraph

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')
ROOT_DIR = './'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# push test
# push test2
def build_model_config(args):
    model_config = {}
    if args.get('use_rxn_class', False):
        atom_fdim = ATOM_FDIM + 10
    else:
        atom_fdim = ATOM_FDIM
    model_config['n_atom_feat'] = atom_fdim
    model_config['n_lg_atom_feat'] = ATOM_FDIM
    if args.get('atom_message', False):
        model_config['n_bond_feat'] = BOND_FDIM
    else:
        model_config['n_bond_feat'] = atom_fdim + BOND_FDIM
    model_config['mpn_size'] = args['mpn_size']
    model_config['mlp_size'] = args['mlp_size']
    model_config['depth'] = args['depth']
    model_config['dropout_mlp'] = args['dropout_mlp']
    model_config['dropout_mpn'] = args['dropout_mpn']
    model_config['atom_message'] = args['atom_message']
    model_config['use_attn'] = args['use_attn']
    model_config['n_heads'] = args['n_heads']
    model_config['t'] = args['t']
    model_config['contrastive_loss'] = args['contrastive_loss']
    model_config['model_type'] = args['model_type']
    return model_config


def save_checkpoint(model, path, epoch, ex=""):
    save_dict = {'state': model.state_dict()}
    if hasattr(model, 'get_saveables'):
        save_dict['saveables'] = model.get_saveables()
    prefix = f'epoch_{epoch + 1}' + ex
    name = prefix + '.pt'
    save_file = os.path.join(path, name)
    torch.save(save_dict, save_file)


def train_epoch(args, epoch, model, train_data, loss_fn, optimizer):
    # torch.cuda.empty_cache()
    model.train()
    train_loss = 0
    train_acc = 0
    print('current epoch: ', epoch)
    for batch_id, batch_data in enumerate(tqdm(train_data)):
        graph_seq_tensors, seq_labels, seq_mask, edit_seq_tensors = batch_data
        seq_mask = seq_mask.to(DEVICE)
        seq_edit_scores, lg_loss = model(graph_seq_tensors, seq_mask, edit_seq_tensors)
        lg_loss.cpu()
        max_seq_len, batch_size = seq_mask.size()
        seq_loss = []

        for idx in range(max_seq_len):
            edit_labels_idx = model.to_device(seq_labels[idx])
            # loss_batch = [seq_mask[idx][i] * loss_fn(seq_edit_scores[idx][i].unsqueeze(0),
            #                                          torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()).sum()
            #               for i in range(batch_size)]
            loss_batch = []
            # valid_loss_cnt = 0
            for i in range(batch_size):
                mask = seq_mask[idx][i]
                assert seq_edit_scores[idx][i].shape == edit_labels_idx[i].shape
                loss = loss_fn(seq_edit_scores[idx][i].unsqueeze(0),
                               torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()).sum()
                loss_batch.append(mask * loss)
                # valid_loss_cnt += mask
                if mask == 1:
                    seq_loss.append(loss)

        total_loss = torch.stack(seq_loss).mean() + lg_loss
        accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)

        train_loss += total_loss.item()
        train_acc += accuracy

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
        # print(total_loss)
        if (batch_id + 1) % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss: %.4f, accuracy: %.4f' % (
                epoch + 1, args['epochs'], batch_id + 1, len(
                    train_data), train_loss / (batch_id + 1), train_acc / (batch_id + 1)), end='', flush=True)

    train_loss = float('%.4f' % (train_loss / len(train_data)))
    train_acc = float('%.4f' % (train_acc / len(train_data)))
    print('\nepoch %d/%d, train loss: %.4f, train accuracy: %.4f' %
          (epoch + 1, args['epochs'], train_loss, train_acc))

    return train_loss, train_acc


def test(model, valid_data, lg_info=None):
    model.eval()
    total_accuracy = 0.0
    first_step_accuracy = 0.0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(tqdm(valid_data)):
            prod_smi_batch, edits_batch, edits_atom_batch, rxn_classes = batch_data
            for idx, prod_smi in enumerate(prod_smi_batch):
                if rxn_classes is None:
                    edits, edits_atom = model.predict(prod_smi)
                else:
                    edits, edits_atom = model.predict(
                        prod_smi, rxn_class=rxn_classes[idx])
                # if judge(edits, edits_batch[idx], edits_atom, edits_atom_batch):
                #     total_accuracy += 1.0
                if edits == edits_batch[idx] and edits_atom == edits_atom_batch[idx]:
                    total_accuracy += 1.0
                if edits[0] == edits_batch[idx][0] and edits_atom[0] == edits_atom_batch[idx][0]:
                    first_step_accuracy += 1.0
    valid_acc = float('%.4f' % (total_accuracy / len(valid_data)))
    valid_first_step_acc = float(
        '%.4f' % (first_step_accuracy / len(valid_data)))

    return valid_acc, valid_first_step_acc


def main(args):
    if args.get('use_rxn_class', False):
        out_dir = os.path.join(ROOT_DIR, 'experiments',
                               args['dataset'], 'with_rxn_class', DATE_TIME)
    else:
        out_dir = os.path.join(ROOT_DIR, 'experiments',
                               args['dataset'], 'without_rxn_class', DATE_TIME)
    os.makedirs(out_dir, exist_ok=True)

    logs_filename = os.path.join(out_dir, 'logs.csv')
    csv_logger = CSVLogger(
        args=args,
        fieldnames=['epoch', 'train_acc', 'valid_acc',
                    'valid_first_step_acc', 'train_loss'],
        filename=logs_filename,
    )

    data_dir = os.path.join(ROOT_DIR, 'data', args['dataset'])
    # load bond, atom and lg vocab
    bond_vocab_file = os.path.join(data_dir, 'train', 'bond_vocab.txt')
    atom_vocab_file = os.path.join(data_dir, 'train', 'atom_lg_vocab.txt')
    lg_vocab_file = os.path.join(data_dir, 'train', 'lg_vocab.txt')
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))
    lg_vocab = Vocab(joblib.load(lg_vocab_file))
    lg_tensors, lg_scopes = process_lg(lg_vocab)
    if args.get('use_rxn_class', False):
        train_dir = os.path.join(data_dir, 'train', 'with_rxn_class')
    else:
        train_dir = os.path.join(data_dir, 'train', 'without_rxn_class')
    eval_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    train_dataset = RetroRandomEditDataset(data_dir='./data/uspto_50k/train/',
                                           use_rxn=args.get('use_rxn_class', False),
                                           max_steps=args['max_steps'], lg_random_aug=args['lg_random_aug'])
    train_data = train_dataset.loader(batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)

    valid_dataset = RetroEvalDataset(
        data_dir=eval_dir, data_file='valid.file.kekulized', use_rxn_class=args['use_rxn_class'])
    test_dataset = RetroEvalDataset(data_dir=test_dir, data_file='test.file.kekulized',
                                    use_rxn_class=args['use_rxn_class'])
    test_data = test_dataset.loader(batch_size=1, num_workers=args['num_workers'])
    valid_data = valid_dataset.loader(
        batch_size=1, num_workers=args['num_workers'])

    model_config = build_model_config(args)
    model = RetroLEE(config=model_config, atom_vocab=atom_vocab,
                     bond_vocab=bond_vocab, tot_vocab=train_dataset.tot_vocab, device=DEVICE, transit_graph=None)
    print(f'Converting model to device: {DEVICE}')
    sys.stdout.flush()
    model.to(DEVICE)
    print("Param Count: ", sum([x.nelement()
                                for x in model.parameters()]) / 10 ** 6, "M")
    print()

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'],
        threshold_mode='abs')

    best_acc = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(
            args, epoch, model, train_data, loss_fn, optimizer)
        valid_acc, valid_first_step_acc = test(model, valid_data)
        scheduler.step(valid_acc)
        print('epoch %d/%d, validation accuracy: %.4f, validation_first_acc: %.4f' %
              (epoch + 1, args['epochs'], valid_acc, valid_first_step_acc))
        print('---------------------------------------------------------')
        print()

        row = {
            'epoch': str(epoch + 1),
            'train_acc': str(train_acc),
            'valid_acc': str(valid_acc),
            'valid_first_step_acc': str(valid_first_step_acc),
            'train_loss': str(train_loss),
        }
        csv_logger.writerow(row)

        if (epoch + 1) > 100 or valid_acc > 0.55:
            save_checkpoint(model, out_dir, epoch, ex="extra")
        # update the best accuracy for saving checkpoints
        if valid_acc >= best_acc:
            print(
                f'Best eval accuracy so far. Saving best model from epoch {epoch + 1} (acc={valid_acc})')
            print('---------------------------------------------------------')
            print()
            save_checkpoint(model, out_dir, epoch)
            best_acc = valid_acc

    csv_logger.close()
    print('Experiment finished!')


def process_lg(lg_vocab):
    lg_size = lg_vocab.size()
    lg_seq = []
    # delete the * notation
    for i in range(lg_size):
        lg_edit = lg_vocab.get_elem(i)
        tmp = build_edit_graph(lg_edit)
        lg_seq.append(tmp)
        # print('hello world')
    # put into the integration graph
    graph_tensors, scopes = get_batch_graphs(lg_seq, use_rxn_class=False)
    return graph_tensors, scopes


def build_edit_graph(edit_to_apply):
    if edit_to_apply[0] != 'Attaching LG':
        return None
    lg_smi = edit_to_apply[1]
    lg_mol = Chem.MolFromSmiles(lg_smi)
    a_id = -1
    for atom in lg_mol.GetAtoms():
        if atom.GetSymbol() == "*":
            a_id = atom.GetIdx()
    rw_lg_mol = Chem.RWMol(lg_mol)
    rw_lg_mol.RemoveAtom(a_id)
    mol_g = MolGraph(mol=rw_lg_mol, rxn_class=False, use_rxn_class=False)
    return mol_g


def set_seed():
    # 必须禁用模型初始化中的任何随机性。
    seed = 233
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed()
    # torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or uspto_full')
    parser.add_argument('--use_rxn_class', default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--atom_message', default=False, action='store_true',
                        help='Node-level or Bond-level message passing')  # bond_level -> d-mpnn
    parser.add_argument('--use_attn', default=False,
                        action='store_true', help='Whether to use global attention')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of heads in Multihead attention')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Maximum number of epochs for training')
    parser.add_argument('--mpn_size', type=int,
                        default=256, help='MPN hidden_dim')
    parser.add_argument('--depth', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--dropout_mpn', type=float,
                        default=0.15, help='MPN dropout rate')
    parser.add_argument('--mlp_size', type=int,
                        default=512, help='MLP hidden_dim')
    parser.add_argument('--dropout_mlp', type=float,
                        default=0.2, help='MLP dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--t', type=float, default=0.05, help='logits scale factor')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which lr will be reduced')
    parser.add_argument('--factor', type=float, default=0.8,
                        help='Factor by which the lr will be reduced')
    parser.add_argument('--thresh', type=float, default=0.01,
                        help='Threshold for measuring the new optimum')
    parser.add_argument('--max_clip', type=int, default=10,
                        help='Maximum number of gradient clip')
    parser.add_argument('--print_every', type=int,
                        default=200, help='Print during train process')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of processes for data loading')
    parser.add_argument('--max_steps', default=9,
                        help='Number of processes for data loading')
    parser.add_argument('--batch_size', default=3, type=int,
                        help='Number of batchsize for data loading')
    parser.add_argument('--lg_random_aug', default=False, action='store_true',
                        help='whther use lg_random_aug')
    parser.add_argument('--contrastive_loss', type=str, default='clip', choices=['clip', 'nxt', 'mse'],
                        help='loss for contrastive learning')
    parser.add_argument('--model_type', type=str, default='gru', choices=['gru', 'gin', 'gcn'],
                        help='loss for contrastive learning')
    args = parser.parse_args().__dict__
    main(args)
