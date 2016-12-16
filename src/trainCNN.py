import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from tweetCNN import tweetCNN
from baseline import get_embeddings_dictionary
from split_hashtag import split_hashtag_to_words

from options import *
    
def glove_per_word(tweets, words, opts):
    embeddings = np.zeros((tweets.shape[0], opts['max_num_words'], opts['embedding_size'], 1))
    for i, tweet in enumerate(tweets['tweet']):
        try:
            split_tweet = tweet.split()
        except:
            continue;
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
            

def trainCNN(tweets, labels):
    
    n_data = tweets['tweet'].shape[0]
    n_valid = 2000
    
    shuffled_indices = np.random.permutation(np.arange(n_data))
    valid_ind = shuffled_indices[:n_valid]
    train_ind = shuffled_indices[n_valid:]   
    
    words = get_embeddings_dictionary()
    
    x_valid = glove_per_word(tweets.loc[valid_ind], words, cnn_params)
    y_valid = labels[valid_ind, :]
    
    
    with tf.Graph().as_default():
        sess = tf.Session()
        
        with sess.as_default():
            cnn = tweetCNN(cnn_params)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            if cnn_params['optimizer']=='Adam':
                train_op = tf.train.AdamOptimizer(cnn_params['lambda']).minimize(cnn.loss, global_step=global_step)
            elif cnn_params['optimizer']=='RMSProp':
                train_op = tf.train.RMSPropOptimizer(cnn_params['lambda'],cnn_params['moment']).minimize(cnn.loss, global_step=global_step)

            # Use timestamps for summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(TF_SAVE_PATH, timestamp))
     
            # Summaries for visualization
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)
            
            train_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            
            valid_summary_op = tf.merge_summary([loss_summary, acc_summary])
            valid_summary_dir = os.path.join(out_dir, 'summaries', 'validation')
            valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, sess.graph)

            # Checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())
            
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                # function to apply backpropagation
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob:cnn_params['dropout_prob']}
                _, step, summaries, loss, accuracy = sess.run(
                                                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                                                        feed_dict
                                                    )
                print('step %d,    loss %.3f,    acc %.2f' %(step,loss,100*accuracy))
                train_summary_writer.add_summary(summaries, step)
                
            def valid_step(x_batch, y_batch, writer=None):
                # function to evaluate performance
                feed_dict = {cnn.x: x_batch, cnn.y: y_batch, cnn.dropout_prob:1.0}
                step, summaries, loss, accuracy,  pred = sess.run(
                                                    [global_step, valid_summary_op, cnn.loss, cnn.accuracy, cnn.y_pred],
                                                    feed_dict
                                                    )
                print('step %d,    loss %.3f,    acc %.2f' %(step,loss,100*accuracy)), 
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

