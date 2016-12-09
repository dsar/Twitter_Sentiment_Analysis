#CHECK dm param in Doc2Vec

from options import *

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

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

def doc2vec():

    sources = {TEST_PREPROC_CACHING_PATH:'TEST',\
               'train_neg.d2v':'TRAIN_NEG',\
               'train_pos.d2v':'TRAIN_POS'}# , 'unsup.txt':'TRAIN_UNS'}

    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(dm=0,dm_concat=1,min_count=1, window=10, size=200, sample=1e-4, negative=5, workers=7, docvecs_mapfile=EMBEDDINGS_FILE_200)

    model.build_vocab(sentences.to_array())

    for epoch in range(WE_params['epochs']):
        logger.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm())

    model.save(DOC2VEC_MODEL_PATH)

    return None, None

