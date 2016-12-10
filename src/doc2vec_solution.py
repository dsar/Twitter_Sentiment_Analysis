#CHECK dm param in Doc2Vec

from options import *

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np
import pandas as pd

# shuffle
from random import shuffle

import os.path

# logging
import logging
import os.path
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def doc2vec(tweets, test_tweets):

    pos = tweets[tweets['sentiment'] == 1]['tweet']
    pos.to_csv(PREPROC_DATA_PATH+'train_pos.d2v', header=False, index=False, encoding='utf-8')
    neg = tweets[tweets['sentiment'] == -1]['tweet']
    neg.to_csv(PREPROC_DATA_PATH+'train_neg.d2v', header=False, index=False, encoding='utf-8')

    test = test_tweets['tweet']
    test.to_csv(PREPROC_DATA_PATH+'test.d2v', header=False, index=False, encoding='utf-8')

    print('POS shape: ',pos.shape)
    print('NEG shape: ',neg.shape)

    model = Doc2Vec(dm=0,dm_concat=1,min_count=1, window=10, size=WE_params['we_features'], sample=1e-4, negative=5, workers=7)#, docvecs_mapfile=EMBEDDINGS_FILE_200)

    if not os.path.exists(DOC2VEC_MODEL_PATH):

        sources = {PREPROC_DATA_PATH + 'test.d2v':'TEST',\
                   PREPROC_DATA_PATH + 'train_neg.d2v':'TRAIN_NEG',\
                   PREPROC_DATA_PATH + 'train_pos.d2v':'TRAIN_POS'}# , 'unsup.txt':'TRAIN_UNS'}

        sentences = LabeledLineSentence(sources)

        model.build_vocab(sentences.to_array())

        for epoch in range(WE_params['epochs']):
            logger.info('Epoch %d' % epoch)
            model.train(sentences.sentences_perm())

        model.save(DOC2VEC_MODEL_PATH)
    else:
        model = model.load(DOC2VEC_MODEL_PATH)

    train_arrays = np.zeros((pos.shape[0] + neg.shape[0], WE_params['we_features']))
    train_labels = np.zeros(pos.shape[0] + neg.shape[0])

    for i in range(pos.shape[0]):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_labels[i] = 1
    for i in range(neg.shape[0]):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i+pos.shape[0]] = model.docvecs[prefix_train_neg]
        train_labels[i+pos.shape[0]] = -1

    # print('train_arrays: ',train_arrays)
    # print('train_labels',train_labels)

    test_arrays = np.zeros((test.shape[0], WE_params['we_features']))
    test_labels = np.zeros(test.shape[0])

    for i in range(test.shape[0]):
        prefix_test_pos = 'TEST_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_labels[i] = 1

    return train_arrays, test_arrays

