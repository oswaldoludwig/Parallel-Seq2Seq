# Parallel seq2seq in TensorFlow
# For further details on the parallel seq2seq model see:
# O. Ludwig, "End-to-end adversarial learning for generative conversational agents" in arXiv:1711.10122, Jan. 2018
# In case of publication using this code, please cite the paper above.                                                                                                                                                     
# Author: Oswaldo Ludwig

import tensorflow as tf
import numpy as np
import os

class operations:
    
    def __init__(self, session):  
        self.session = session

    def get_graph(self):
        new_saver = tf.train.import_meta_graph('./PS2S.meta')
        new_saver.restore(self.session,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name('PT2T_input:0')
        output_context = graph.get_tensor_by_name('PT2T_output_context:0')
        predicted = graph.get_tensor_by_name('PT2T_predicted:0')
        target = graph.get_tensor_by_name('PT2T_target:0')
        cost = graph.get_tensor_by_name('PT2T_cost:0')
        learning_rate = graph.get_tensor_by_name('PT2T_learning_rate:0')
        optimizer = graph.get_operation_by_name('PS2S_optimizer')
        self.input = input
        self.output_context = output_context
        self.target = target
        self.cost = cost
        self.optimizer = optimizer
        self.predicted = predicted
        self.saver = new_saver
        self.learning_rate = learning_rate

    def train_ps2s(self, xt, yt, Learning_rate):
        length_out = yt.shape[1]
        Cost = 0
        for t in range(length_out - 1):  #  loop over the output/target sequence
            feed = {self.input: np.asarray(xt), self.output_context: np.asarray(yt[:,0:t+1,:]), self.target: np.asarray(yt[:,t+1,:]), self.learning_rate: Learning_rate}
            Batch_cost,_ = self.session.run([self.cost, self.optimizer], feed)
            Cost += Batch_cost
        return Cost
        
    def greedy_decoder(self, xt, BOS_vector, EOS_vector):
        pred = BOS_vector
        bs = xt.shape[0]
        od = EOS_vector.shape
        Y = []
        Y.append(BOS_vector)
        Yany = np.asarray(Y)
        Y = np.asarray(np.expand_dims(Y, axis=0))
        feed = {self.input: np.asarray(xt), self.output_context: Y, self.target: Yany}
        count = 0
        while (np.argmax(pred) != np.argmax(EOS_vector)) and (count < 120):
            pred = self.session.run([self.predicted], feed)
            Y = np.concatenate((Y, pred), axis=1)
            feed = {self.input: np.asarray(xt), self.output_context: Y, self.target: Yany}
            count += 1
        return Y
    def save_ps2s(self):
        self.saver.save(self.session, './PS2S')
        print('Session saved')
        

class initialize:
    
    def __init__(self, n_layers_EI, n_layers_EO, dim_EI, dim_EO, dim_input, dim_output, input_seq_len):    
        self.n_layers_EI = n_layers_EI   
        self.n_layers_EO = n_layers_EO
        self.dim_EI = dim_EI
        self.dim_EO = dim_EO
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.input_seq_len = input_seq_len

    def setup_ps2s(self):
        
        def ps2s(self, x, y_partial):
            # This implements the TF graph of the parallel seq2seq to be used in any application
            
            sy = tf.shape(y_partial) 
            seq_len = sy[1]
            
            cells_EI = []                                                                       
            for _ in range(self.n_layers_EI):                                                        
                cell_EI = tf.contrib.rnn.LSTMCell(self.dim_EI)                                       
                cells_EI.append(cell_EI)                                                        
                
            stack_EI = tf.contrib.rnn.MultiRNNCell(cells_EI)
            
            cells_EO = []
            for _ in range(self.n_layers_EO):
                cell_EO = tf.contrib.rnn.LSTMCell(self.dim_EO)
                cells_EO.append(cell_EO)                 
                
            stack_EO = tf.contrib.rnn.MultiRNNCell(cells_EO)
                
            with tf.variable_scope("input1") as scope:
                EI, _ = tf.nn.dynamic_rnn(stack_EI, x, dtype=tf.float32)

            mean_EI = tf.math.reduce_mean(EI, axis=1, name = 'encoder_input')

            with tf.variable_scope("input2") as scope:
                EO, _ = tf.nn.dynamic_rnn(stack_EO, y_partial, dtype=tf.float32)
         
            last_EO = tf.gather(EO, seq_len - 1, axis = 1, name = 'encoder_output')
            
            E = tf.concat([mean_EI, last_EO], 1)

            W = tf.Variable(tf.truncated_normal([(self.dim_EI + self.dim_EO), self.dim_output], stddev=0.01))                                                                                                                                                           
            b = tf.Variable(tf.constant(0.5, shape=[self.dim_output]))
           
            outputs = tf.nn.softmax(tf.matmul(E, W) + b, axis=1)

            return outputs
        
        PT2T = tf.Graph()
        with PT2T.as_default():                        
            input = tf.placeholder(tf.float32, [None, None, self.dim_input], name='PT2T_input')
            print('defining the output placeholder')                                     
            output_context = tf.placeholder(tf.float32, [None, None, self.dim_output], name='PT2T_output_context')                            
            print('defined')
            target = tf.placeholder(tf.float32, [None, self.dim_output], name='PT2T_target')
            learning_rate = tf.placeholder(tf.float32,shape=(),name='PT2T_learning_rate')                                            
            predicted = ps2s(self, input, output_context)
            tf.identity(predicted, name='PT2T_predicted')
            loss = - tf.reduce_mean(tf.math.multiply(target, tf.log(predicted + 1e-5)),axis=1)  #  this implements the categorical cross-entropy loss 
            cost = tf.reduce_mean(loss, axis=0, name='PT2T_cost')                                              
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='PS2S_optimizer')
            saver = tf.train.Saver(name='PT2T_saver')
    
        with tf.Session(graph=PT2T) as session:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            if os.path.isfile('PS2S.meta'):
                ans = raw_input('\n\n\n\n A parallel seq2seq model already exists, do you want to overwrite (o) or resume training (r)? \n')
                if ans == 'o':
                    saver.save(session, './PS2S')
            else:
                saver.save(session, './PS2S')
