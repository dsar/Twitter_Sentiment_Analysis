import tensorflow as tf

class tweetCNN(object):
    """
    DESCRIPTION: 
        creates a class which generates a tensorflow graph for classification of text data with 
        convolutional neural networks
        Initialization is achieved via a dictionary (opts)
    """
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
                Wname = 'W_conv_%s' % fsize
                W = tf.get_variable(Wname, shape=[fsize, self.embedding_size, 1 ,self.n_filters],
                                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.2, shape=[self.n_filters]),
                                name='b')
                
                pre_act = tf.nn.bias_add(tf.nn.conv2d(self.x, W, strides=[1,1,1,1],padding='VALID',name='conv'),
                                            b, name='pre_activation')                                        
                act = tf.nn.relu(pre_act, name='activation')
                
                pooled = tf.nn.max_pool(act,
                                        ksize=[1,max_num_words-fsize+1, 1, 1],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='max_pooled')
                
                out.append(pooled)
                
        self.features = tf.reshape(tf.concat(3, out),
                                    [-1, n_filters_total], name='all_features')
                                        
        with tf.name_scope('dropout'):
            self.features_dropout = tf.nn.dropout(self.features, self.dropout_prob, name='after_dropout_1')
            softmax_input = self.features_dropout
            n_in_softmax = n_filters_total
            print(n_filters_total)
        
        if opts['n_layers']==2:
            with tf.name_scope('hidden_layer'):
                W = tf.get_variable("W_hidden", shape=[n_filters_total, opts['n_hidden']],
                                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.2, shape=[opts['n_hidden']]), name='b_hidden')
                pre_act = tf.nn.xw_plus_b(self.features_dropout, W, b, name='pre_activation')
                act = tf.nn.relu(pre_act, name='activation')               
                
            with tf.name_scope('dropout'):
                softmax_input = tf.nn.dropout(act, self.dropout_prob, name='after_dropout_2')
                n_in_softmax = opts['n_hidden']

        with tf.name_scope('softmax'):
            W = tf.get_variable('W_softmax', shape=[n_in_softmax, self.n_classes], 
                                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name='b_softmax')
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(softmax_input, W, b), name='prediction_probabilites')
            self.y_pred = tf.argmax(self.scores,  axis=1,  name = 'predicted_classes')
        
        with tf.name_scope('loss_calculation'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y), name='loss')
            
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.y_pred, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')

