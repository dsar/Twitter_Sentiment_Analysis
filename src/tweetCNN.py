import tensorflow as tf

class tweetCNN(object):
    def __init__(self, opts):
                            
        self.embedding_size = opts['embedding_size']
        self.n_filters = opts['n_filters']
        self.filter_sizes = opts['filter_sizes']
        self.n_classes = opts['n_classes']
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        
        max_num_words = opts['max_num_words']
              
        n_filters_total = self.n_filters*len(self.filter_sizes)
        
        self.x = tf.placeholder(tf.float32, [None, max_num_words, self.embedding_size,  1], name='embedding') 
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], name='class_probability')        	
        
        out = []
        for i, fsize in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % fsize):
                W = tf.Variable(tf.truncated_normal([fsize, self.embedding_size, 1 ,self.n_filters],
                                                                        stddev = 0.1),
                                name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]),
                                name='b')
                
                pre_act = tf.nn.bias_add(tf.nn.conv2d(self.x, W, strides=[1,1,1,1],padding='VALID',name='conv'),b)                                        
                act = tf.nn.relu(pre_act, name='nonlinearity')
                
                pooled = tf.nn.max_pool(act,
                                        ksize=[1,max_num_words-fsize+1, 1, 1],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='max_pool')
                
                out.append(pooled)
                
        self.features = tf.reshape(tf.concat(3, out),
                                    [-1, n_filters_total])
                                        
        with tf.name_scope('dropout'):
            self.features_dropout = tf.nn.dropout(self.features, self.dropout_prob)
         
        with tf.name_scope('prediction'):
            W = tf.Variable(tf.truncated_normal([n_filters_total, self.n_classes], stddev=0.1), 
                            name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name='b')
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.features_dropout, W, b), name='prediction_probabilites')
            self.y_pred = tf.argmax(self.scores,  axis=1,  name = 'predictions')
        
        with tf.name_scope('loss_function'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y))
            
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.y_pred, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
