import tensorflow as tf
import numpy as np

from tweetCNN import tweetCNN
from trainCNN import glove_per_word
from baseline import get_embeddings_dictionary

from options import *

def evalCNN(test_tweets, path):
    
    print('getting dictionary for embeddings...')
    words = get_embeddings_dictionary()        
    print('embeddings for test data...')
    test_reptweets = glove_per_word(test_tweets,  words,  cnn_params)
    print('glove embeddings are obtained successfully for test data \n')
    
    graph = tf.Graph()
    with graph.as_default():
        if cnn_params['save_from_file']:
            checkpoint_file = tf.train.latest_checkpoint(cnn_params['checkpoint_dir'])
        else:
            checkpoint_file = path
            
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            x = graph.get_operation_by_name("embedding").outputs[0]
            dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]
            predictions = graph.get_operation_by_name("prediction/predictions").outputs[0]
            
            test_predictions = sess.run(predictions, {x:test_reptweets,  dropout_prob:1.0})
            test_predictions[test_predictions==0] = -1
            
    return test_predictions
