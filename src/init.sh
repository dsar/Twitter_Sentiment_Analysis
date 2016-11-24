#!/bin/bash

DATA_PATH='../data/'
POS_FILE=$DATA_PATH'train_pos.txt'
NEG_FILE=$DATA_PATH'train_neg.txt'

echo -n 'build vocabulary... '
bash build_vocab.sh $POS_FILE $NEG_FILE
echo 'DONE'
echo -n 'cut vocabulary...'
bash cut_vocab.sh
echo 'DONE'
echo -n 'pickle vocabulary...'
python3 pickle_vocab.py
echo 'DONE'
echo 'build coocurence matrix...'
python3 cooc.py
echo 'DONE'
echo -n 'build word embeddings...'
python3 glove_solution.py
echo 'DONE'
