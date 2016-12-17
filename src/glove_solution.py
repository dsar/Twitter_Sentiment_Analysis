#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
from build_embeddings import store_embeddings_to_txt_file
from options import *



def main():
    print("loading cooccurrence matrix")
    with open(COOC_FILE, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = algorithm['options']['WE']['we_features']
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    # epochs = 10

    for epoch in range(algorithm['options']['WE']['epochs']):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    # np.save(GLOVE_DATA_PATH+'embeddings', xs)
    words = {} #key= word, value=embeddings
    we = xs
    print('we shape', we.shape)
    vocab_file = open(VOCAB_CUT_FILE, "r")
    for i, line in enumerate(vocab_file):
        words[line.rstrip()] = we[i]
    store_embeddings_to_txt_file(words, MY_EMBEDDINGS_TXT_FILE)



if __name__ == '__main__':
    main()
