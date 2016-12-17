#!/usr/bin/env python3
import pickle
from options import *

def main():
    vocab = dict()
    with open(VOCAB_CUT_FILE) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
