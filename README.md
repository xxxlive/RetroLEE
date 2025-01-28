# RetroLEE
## Environment Requirements
```
conda create -n retrolee python=3.10 \
conda activate retrolee \
pip install rdkit \
pip install torch
```
## Data preprocessing
1) generate the edit labels and the edits sequence for reaction
```
python preprocess.py --mode train \
python preprocess.py --mode valid \
python preprocess.py --mode test \
```
2) prepare the data for training
```
python prepare_data.py
```
## Train RetroLEE model
Go to the RetroLEE folder and run the following to train the model with specified dataset (default: USPTO_50k)
```
python train_retroLEE.py --dataset uspto_50k --use_rxn_class False
```
The trained model will be saved at RetroLEE/experiments/uspto_50k/without_rxn_class/
## Evaluate using a trained model
To evaluate the trained model, run
```
python multi_eval.py --experiments checkpoint_folder
you may need change the epoch arrary 
for example epcochs = ['120']
```
to get the raw prediction file saved at RetroLEE/experiments/.../pred_results.txt and csv file
## Reproducing
we give the checkpoint and result file under experiments/uspto_50k

## Reference
Our model used code of previous work Graph2Edits.
