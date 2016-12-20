import os
import time

import tensorflow as tf
import numpy as np

from tweetCNN import tweetCNN
from build_embeddings import get_embeddings_dictionary
from split_hashtag import split_hashtag_to_words

from options import *
    
def glove_per_word(tweets, words, opts):
    """
    DESCRIPTION: 
        outputs a matrix which corresponds to the embeddings of each word in each tweet
    INPUT:
        tweets: a Dataframe which contains a set of tweets with a keyword 'tweet', tweets['tweet']
        words: dictionary for embeddings
        opts: a dictionary 
    OUTPUT:
        embeddings: a matrix by [# of input tweets]x[max # words in a tweet]x[embedding dimension] 
        each row of which corresponds to the embedding of a word in a tweet
    """
    embeddings = np.zeros((tweets.shape[0], opts['max_num_words'], opts['embedding_size'], 1))
    for i, tweet in enumerate(tweets['tweet']):
        try:
            split_tweet = tweet.split()
        except:
            continue;
            
        # keed two index variables
        # one for words we obtained after splitting, but splitted words can be splitted into tokens
        # so the other one for keeping the record of embeddings with splitted and unsplitted words        
        ind_word = 0
        ind_embed = 0
        
        for k in range(opts['max_num_words']):
            if k<len(split_tweet):
                word = split_tweet[ind_word]
                try:
                    embeddings[i, ind_embed, :, :] = words[word].reshape((1,1,-1,1))
                    ind_word+=1
                    ind_embed+=1
                    
                except:
                    if (not word.startswith("#")):
                        word = "#" + word
                    tokens=split_hashtag_to_words(word)
                    for token in tokens.split():
                        if((len(token) != 1) or (token == "a") or (token == "i")):
                            try:
                                embeddings[i, ind_embed, :, :] = words[token].reshape((1,1,-1,1))
                                ind_embed += 1
                            except:
                                continue;
                    ind_word += 1
                    continue;
    return embeddings


def batch_iter(train_indices, batch_size, shuffle=True):
	"""
	DESCRIPTION: 
		batch iterator
	INPUT:
		train_indices: an array of numbers
		batch_size: a constant
		shuffle: optional flag
	"""
	n_ind = len(train_indices)
	if shuffle:
		shuffled_indices = np.random.permutation(train_indices)
	else:
		shuffled_indices = train_indices
	for batch_num in range(int(np.ceil(n_ind/batch_size))):
		start_index = batch_num * batch_size
		end_index = min((batch_num + 1) * batch_size, n_ind)
		if start_index != end_index:
			yield shuffled_indices[start_index:end_index]


def prepare_data(tweets, labels, cnn_params):
    """
    DESCRIPTION:
        shuffles the data, choses a validation dataset and obtains the 
        dictionary for word embeddings
    INPUT:
        tweets: a Dataframe which contains a set of tweets with a keyword 'tweet', tweets['tweet']
        labels: nx2 label matrix where each row corresponds to the the label of data to the class of
        corresponding col, n is the number of tweets 
        cnn_params: a dictionary
    OUTPUT:
        train_ind: shuffled indices for training dataset
        x_valid: an array of size [# of valid. tweets]x[max # words in a tweet]x[embedding dimension]
        y_valid: an label array of size [# of valid. tweets]x2
    """
    n_data = tweets['tweet'].shape[0]
    
    shuffled_indices = np.random.permutation(np.arange(n_data))
    valid_ind = shuffled_indices[:cnn_params['n_valid']]
    train_ind = shuffled_indices[cnn_params['n_valid']:]   
    
    words = get_embeddings_dictionary()
    
    x_valid = glove_per_word(tweets.loc[valid_ind], words, cnn_params)
    y_valid = labels[valid_ind, :]
    
    return train_ind, x_valid, y_valid,  words
    

def trainCNN(tweets, labels, cnn_params):
    """
    DESCRIPTION:
        trains a CNN, an instance of tweetCNN object, for tweet embeddings from sctratch
    INPUT:
        tweets: a Dataframe which contains a set of tweets with a keyword 'tweet', tweets['tweet']
        labels:  nx2 label matrix where each row corresponds to the the label of data to the class of
        corresponding col, n is the number of tweets 
        cnn_params: a dictionary
    OUTPUT: 
        path: a file name (with path) that indicates the last saved checkpoint for tweetCNN instance        
    """
    
    train_ind, x_valid, y_valid, words = prepare_data(tweets,  labels, cnn_params)
    
    with tf.Graph().as_default():
        sess = tf.Session()
        
        with sess.as_default():
            cnn = tweetCNN(cnn_params)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            
            learning_rate = tf.train.exponential_decay(cnn_params['lambda'], global_step, 
                                                        cnn_params['lambda_decay_period'], cnn_params['lambda_decay_rate'], 
                                                        staircase=True, name='learning_rate')
            if cnn_params['optimizer']=='Adam':
                train_op = tf.train.AdamOptimizer(learning_rate, name='optimizer').minimize(cnn.loss, global_step=global_step, name='optim_operation')
            elif cnn_params['optimizer']=='RMSProp':
                train_op = tf.train.RMSPropOptimizer(learning_rate,cnn_params['moment'], name='optimizer').minimize(cnn.loss, global_step=global_step, name='optim_operation')

            # Use timestamps for summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(TF_SAVE_PATH, timestamp))
     
            # Summaries for visualization
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)
            lambda_summary = tf.summary.scalar('learning_rate', learning_rate)
            
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, lambda_summary], name='training_summaries')
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            
            valid_summary_op = tf.merge_summary([loss_summary, acc_summary], name='validation_summaries')
            valid_summary_dir = os.path.join(out_dir, 'summaries', 'validation')
            valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, sess.graph)

            # Checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=cnn_params['n_checkpoints_to_keep'], name='saver')
            
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                # function to apply backpropagation
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob:cnn_params['dropout_prob']}
                _, step, summaries, loss, accuracy = sess.run(
                                                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                                                        feed_dict
                                                    )
                print('step %d,    loss %.3f,    accuracy %.2f' %(step,loss,100*accuracy))
                train_summary_writer.add_summary(summaries, step)
                
            def valid_step(x_batch, y_batch, writer=None):
                # function to evaluate performance
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob:1.0}
                step, summaries, loss, accuracy,  pred = sess.run(
                                                    [global_step, valid_summary_op, cnn.loss, cnn.accuracy, cnn.y_pred],
                                                    feed_dict
                                                    )
                print('step %d,    loss %.3f,    accuracy %.2f' %(step,loss,100*accuracy)), 
                if writer:
                    writer.add_summary(summaries, step)
                

            for ep in range(cnn_params['n_epochs']):
                for batch_ind in batch_iter(train_ind, cnn_params['batch_size']):
                    minibatch_x = glove_per_word(tweets.loc[batch_ind], words, cnn_params)                    
                    minibatch_y = labels[batch_ind, :]
                    
                    train_step(minibatch_x, minibatch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % cnn_params['eval_every'] == 0:
                        print("\nEvaluation:")
                        valid_step(x_valid, y_valid, writer=valid_summary_writer)
                    if current_step % cnn_params['checkpoint_every'] == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        
    return path

def trainCNN_fromcheckpoint(tweets, labels, cnn_params):
    """
    DESCRIPTION:
        continues to train a CNN, an instance of tweetCNN object, for tweet embeddings from a checkpoint,
        i.e., restores a previous tensorflow graph and uses same parameters
    INPUT:
        tweets: a Dataframe which contains a set of tweets with a keyword 'tweet', tweets['tweet']
        labels:  nx2 label matrix where each row corresponds to the the label of data to the class of
        corresponding col, n is the number of tweets 
        cnn_params: a dictionary
    OUTPUT: 
        path: a file name (with path) that indicates the last saved checkpoint for tweetCNN instance        
    """
    train_ind, x_valid, y_valid,  words = prepare_data(tweets,  labels, cnn_params)
    
    graph = tf.Graph()
    with graph.as_default():
        checkpoint_file = tf.train.latest_checkpoint(cnn_params['checkpoint_dir'])
            
        sess = tf.Session()
        with sess.as_default():
            
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            
            x = graph.get_operation_by_name('embedding').outputs[0]
            y = graph.get_operation_by_name('class_probability').outputs[0]
            
            dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]     
            
            predictions = graph.get_operation_by_name('softmax/predicted_classes').outputs[0]
            loss = graph.get_operation_by_name('loss_calculation/loss').outputs[0]
            accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]            
            
            global_step = graph.get_operation_by_name('global_step').outputs[0]
            learning_rate = graph.get_operation_by_name('learning_rate').outputs[0]
            train_op = graph.get_operation_by_name('optim_operation').outputs[0]
            
            checkpoint_dir = cnn_params['checkpoint_dir']
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            out_dir, _ = os.path.split(checkpoint_dir)
            
            # Summaries
            loss_summary = tf.summary.scalar('loss', loss)
            acc_summary = tf.summary.scalar('accuracy', accuracy)
            lambda_summary = tf.summary.scalar('learning_rate', learning_rate)
            
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, lambda_summary], name='training_summaries')
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            
            valid_summary_op = tf.summary.merge([loss_summary, acc_summary], name='validation_summaries')
            valid_summary_dir = os.path.join(out_dir, 'summaries', 'validation')
            valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, sess.graph)
                        
            saver = tf.train.Saver(tf.all_variables())
            
            uninitialized_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            sess.run(tf.initialize_variables(uninitialized_vars))
            
            def train_step(x_batch, y_batch):
                feed_dict = {x: x_batch, y: y_batch, dropout_prob:cnn_params['dropout_prob']}
                _, step, summaries, loss_value, accuracy_value = sess.run(
                                                        [train_op, global_step, train_summary_op, loss, accuracy],
                                                        feed_dict
                                                    )
                print('step %d,    loss %.3f,    acc %.2f' %(step,loss_value,100*accuracy_value))
                train_summary_writer.add_summary(summaries, step)
                
            def valid_step(x_batch, y_batch, writer=None):
                feed_dict = {x: x_batch, y: y_batch, dropout_prob:1.0}
                step, summaries, loss_, accuracy_,  pred = sess.run(
                                                    [global_step, valid_summary_op, loss, accuracy, predictions],
                                                    feed_dict
                                                    )
                print('step %d,    loss %.3f,    acc %.2f' %(step,loss_,100*accuracy_)), 
                if writer:
                    writer.add_summary(summaries, step)
                

            for ep in range(cnn_params['n_epochs']):
                for batch_ind in batch_iter(train_ind, cnn_params['batch_size']):
                    
                    minibatch_x = glove_per_word(tweets.loc[batch_ind], words, cnn_params)                    
                    minibatch_y = labels[batch_ind, :]
                    
                    train_step(minibatch_x, minibatch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % cnn_params['eval_every'] == 0:
                        print("\nEvaluation:")
                        valid_step(x_valid, y_valid, writer=valid_summary_writer)
                    if current_step% cnn_params['checkpoint_every'] == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        

    return path

