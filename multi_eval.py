import numpy as np
import pandas
import pandas as pd
import os
import argparse
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
import torch
from rdkit import Chem, RDLogger

from models.damn_search import DamnSearch
from models.retrolee import RetroLEE
from utils.rxn_graphs import Vocab

lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('no mol', flush=True)
        return smi
    if mol is None:
        return smi
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def canonicalize_p(smi):
    p = canonicalize(smi)
    p_mol = Chem.MolFromSmiles(p)
    [a.SetAtomMapNum(a.GetIdx() + 1) for a in p_mol.GetAtoms()]
    p_smi = Chem.MolToSmiles(p_mol)
    return p_smi


def sig_eval(args, test_file, exp_dir, epoch, pred_file, df_file):
    test_data = joblib.load(test_file)
    checkpoint = torch.load(os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch)), map_location=DEVICE)
    config = checkpoint['saveables']
    atom_vocab = config['atom_vocab']
    bond_vocab = config['bond_vocab']
    tot_vocab = Vocab(atom_vocab.elem_list + bond_vocab.elem_list + ['Terminate', 'Initialize', 'Null'])
    model = RetroLEE(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()
    pred_res = []
    top_k = np.zeros(args.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = DamnSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))

    with open(pred_file, 'a') as fp:
        fp.write('current epoch {}'.format(epoch))
        for idx in p_bar:
            tmp_predictions = []
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class
            edit_steps = len(rxn_data.edits)
            counter.append(edit_steps)
            r, p = rxn_smi.split('>>')
            r_mol = Chem.MolFromSmiles(r)
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
            r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
            r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)
            r_set = set(r_smi.split('.'))
            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)
            fp.write(f'({idx}) {rxn_smi}\n')
            beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                correct = pred_set == r_set
                str_edits = '|'.join(f'({str(edit)};{p})' for edit, p in zip(
                    path['rxn_actions'], path['edits_prob']))
                fp.write(
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')
                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                tmp_predictions.append(pred_smi)
            while len(tmp_predictions) < 10:
                tmp_predictions.append('final_smi_unmapped')
            assert len(tmp_predictions) == 10

            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)

            for edit in rxn_data.edits:
                if edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (
                        2, 2) or edit[1] == (2, 3):
                    stereo_rxn.append(idx)
                    if beam_matched:
                        stereo_rxn_cor.append(idx)

            msg = 'average score'
            for beam_idx in [1, 3, 5, 10]:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.4f' % (beam_idx, match_acc)
            p_bar.set_description(msg)
            pred_res.append(
                {'product': p, 'reactant': r_smi, 'predictions': '\t'.join(tmp_predictions), 'rxn_class': rxn_class,
                 'tgt_edit': rxn_data.edits, 'rxn_step': len(rxn_data.edits), 'rxn_id': rxn_data.rxn_id})

        edit_steps = Counter(counter)
        edit_steps_correct = Counter(edit_steps_cor)
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')
        fp.write(
            f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))
        df = pd.DataFrame(pred_res)
        for beam_idx in [1, 3, 5, 10]:
            match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
            msg += ', t%d: %.4f' % (beam_idx, match_acc)
            s = f"t{beam_idx}"
            df[s] = match_acc
        df.to_csv(df_file)


def eval(args, epochs):
    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    if args.use_rxn_class:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
    else:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')
    txt_file_names = []
    csv_file_names = []
    file_num = 1
    pred_file = None
    df_file = None
    while True:
        pred_file = os.path.join(exp_dir, f'pred_results_{file_num}.txt')
        df_file = os.path.join(exp_dir, f'result_{file_num}.csv')
        if not os.path.exists(pred_file): break
        file_num += 1

    for item in epochs:
        txt_file_names.append(os.path.join(exp_dir, f'pred_results_{file_num}.txt'))
        csv_file_names.append(os.path.join(exp_dir, f'result_{file_num}_{item}.csv'))
        file_num += 1
    p_obj = Parallel(n_jobs=5)
    p_obj(delayed(sig_eval)(args, test_file, exp_dir, epochs[i], txt_file_names[i], csv_file_names[i]) for i in
          range(len(epochs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k',
                        help='dataset: USPTO_50k or USPTO_full')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='27-06-2022--10-27-22',
                        help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    # epochs = [f"{i}extra" for i in range(101, 200)]
    epochs = ['130']
    eval(args, epochs)


if __name__ == '__main__':
    main()
