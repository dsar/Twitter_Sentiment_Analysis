#!/usr/bin/env python3
import pickle
from options import *

def main():
    vocab = dict()
    with open(DATA_PATH+'vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(DATA_PATH+'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
