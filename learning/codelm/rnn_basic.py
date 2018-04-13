#classifying if it's an opcode or not

#framework

#extract_and_prepare
#generate_datasets
#generate_batch
#generate_model
#train_model
#test_model
import tensorflow as tf
import numpy as np
import utilities as ut
import random  
import word2vec as w2v

class Data(object):
    
    def __init__(self):
        self.percentage = 80
        self.embedder = w2v.Word2Vec()
        self.max_time = 30

    def get_embedding(self, token_data, sym_dict, mem_start):
        
        dict_data, _, dictionary, reverse_dictionary = self.embedder.build_dataset(token_data)
        final_embeddings = self.embedder.train_skipgram(dict_data, len(dictionary), reverse_dictionary, sym_dict, mem_start)
        return dict_data, dictionary, final_embeddings

    def extract_and_prepare_data(self,cnx,format):

        raw_data = ut.get_data(cnx,format,['span'])
        token_data = list()
        for row in raw_data:
            token_data.extend(row[0])
        print len(token_data)

        offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
        sym_dict, mem_start = ut.get_sym_dict(offsets_filename)
        offsets = ut.read_offsets(offsets_filename)

        dict_data, dictionary, final_embeddings = self.get_embedding(token_data, sym_dict, mem_start)
            
        embedding_size = self.embedder.embedding_size
    
        #create the dataset
        #x's are the original instructions
        #y can be timing or any global property

        #create the entire dataset from the learnt embeddings
        self.x = np.zeros(shape = [len(raw_data),self.max_time,embedding_size]) 
        self.y = np.zeros(shape = [len(raw_data),1])
        self.lengths = np.zeros(shape = [len(raw_data)])
        self.num = 0

        for i,row in enumerate(raw_data):
            bb = row[0]

            cur_bb = np.ndarray(shape = [self.max_time, embedding_size])
            cur_ins = np.ndarray(shape = [embedding_size])
            time = 0

            start = False
            overflow = False

            for token in bb:
                if token >= offsets[0] and token < offsets[4]:
                    if start == True and time < self.max_time:
                        cur_bb[time,:] = cur_ins
                        time += 1
                    cur_ins = final_embeddings[dictionary.get(token,0)]
                    if time == self.max_time:
                        overflow = True
                        break
                    start = True
                else:
                    cur_ins += final_embeddings[dictionary.get(token,0)]
                    
            if not overflow and start:
                cur_bb[time,:] = cur_ins
                self.x[i,:,:] = cur_bb
                self.y[i,0] = row[1]
                self.lengths[i] = time
                self.num += 1

        print "total filtered : " + str(self.num)

        self.x = self.x[:self.num,:,:]
        self.y = self.y[:self.num]
        self.lengths = self.lengths[:self.num]

    def generate_datasets(self):
        assert self.x.shape[0] == self.y.shape[0]
        size = self.y.shape[0]
        split = (size * self.percentage) // 100
        self.train_x  = self.x[:split,:,:]
        self.train_y = self.y[:split]
        self.train_lengths = self.lengths[:split]
        self.test_x = self.x[(split + 1):,:,:]
        self.test_y = self.y[(split + 1):]
        self.test_lengths = self.lengths[(split + 1):]
        
    def generate_batch(self, batch_size):
        population = range(self.train_x.shape[0])
        embedding_size = self.embedder.embedding_size
        selected = random.sample(population,batch_size)
        batch_x = np.ndarray(shape = [batch_size,self.max_time,embedding_size])
        batch_y = np.ndarray(shape = [batch_size,1])
        batch_lengths = np.ndarray(shape = [batch_size])
        for i,index in enumerate(selected):
            batch_x[i] = self.train_x[index,:,:]
            batch_y[i] = self.train_y[index]
            batch_lengths[i] = self.train_lengths[index]

        

        return batch_x, batch_y, batch_lengths
  
    
class Model(object):

    def __init__(self, data):
        self.data = data
        self.learning_rate = 1.0
        self.epochs = 10
        self.batch_size = 1000
        self.hidden_size = 128

    def generate_model(self):

        embedding_size = self.data.embedder.embedding_size
        learning_rate = self.learning_rate
        max_time = self.data.max_time
        
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.x = tf.placeholder(tf.float32, shape = [self.batch_size,max_time,embedding_size])
            self.y = tf.placeholder(tf.float32, shape = [self.batch_size,1])
            self.lengths = tf.placeholder(tf.int32, shape = [self.batch_size])

            #let's not give the hidden state as input at first - for each batch of training this will start at an arbitrary or a zero hidden state
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            output, state = tf.nn.dynamic_rnn(cell=cell, inputs=self.x, time_major = False, dtype=tf.float32, sequence_length=self.lengths)

            #the output of the final layer will be taken
            #output shape = [batch_size, max_time, hidden_size]
            #self.last_state = tf.reshape(tf.slice(state,[0,max_time,0],[-1,1,-1]),[self.batch_size,-1])
            self.last_state = state[1]

            #fully connected layer
            W = tf.Variable(tf.random_normal(shape = [self.hidden_size, 1], dtype = tf.float32))
            b = tf.Variable(tf.random_normal(shape = [1], dtype = tf.float32))
        
            num_output = tf.add(tf.matmul(self.last_state,W),b)
            self.loss = tf.nn.l2_loss(num_output - self.y)
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)

            self.init_op = tf.global_variables_initializer()
        
           
    def train_model(self):

        epochs = self.epochs

        with tf.Session(graph=self.graph) as sess:
            # initialise the variables
            writer = tf.summary.FileWriter('/tmp/tensorflow/', graph=tf.get_default_graph())

            sess.run(self.init_op)
            total_batch = int(self.data.train_x.shape[0]) / self.batch_size
            for epoch in range(self.epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y, batch_lengths = self.data.generate_batch(self.batch_size)
                    _, c = sess.run([self.optimizer, self.loss], 
                                    feed_dict={self.x: batch_x, self.y: batch_y, self.lengths: batch_lengths})
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
      
    def test_model(self,params):
        W,b = params
        with tf.Session(graph=self.graph) as sess:
            print(sess.run([self.accuracy, self.final_output], feed_dict={self.x: self.data.test_x, self.y_raw: self.data.test_y, self.W: W, self.b: b}))
 
        print self.data.test_y



