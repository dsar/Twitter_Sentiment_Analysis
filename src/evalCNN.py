from options import *

import tensorflow as tf

from trainCNN import glove_per_word
from build_embeddings import get_embeddings_dictionary

def evalCNN(test_tweets, path, cnn_params):
    """
    DESCRIPTION:
        generates the predictions of a test datatset using a trained CNN
    INPUT:
        test_tweets: a Dataframe which contains a set of tweets with a keyword 'tweet', tweets['tweet']
        path: a file name (with path) that indicates the last saved checkpoint for tweetCNN instance 
        cnn_params: a dictionary
    OUTPUT: 
        an array of size [# of tweets in test dataset]x1, where the values {-1,1} indicates the classes      
    """    
    
    print('getting dictionary for embeddings...')
    words = get_embeddings_dictionary()        
    print('embeddings for test data...')
    test_reptweets = glove_per_word(test_tweets,  words,  cnn_params)
    print('glove embeddings are obtained successfully for test data \n')
    
    graph = tf.Graph()
    with graph.as_default():
        if cnn_params['save_from_file']:
            checkpoint_file = tf.train.latest_checkpoint(path)
        else:
            checkpoint_file = path
            
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            x = graph.get_operation_by_name("embedding").outputs[0]
            dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]
            predictions = graph.get_operation_by_name("softmax/predicted_classes").outputs[0]
            
            test_predictions = sess.run(predictions, {x:test_reptweets,  dropout_prob:1.0})
            test_predictions[test_predictions==0] = -1
            
    return test_predictions

