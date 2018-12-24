#main data file

import numpy as np
import common_libs.utilities as ut
import random
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


class Data(object):

    """
    Main data object which extracts data from a database, partition it and gives out batches.

    """


    def __init__(self, data=None): #copy constructor

        self.percentage = 80
        self.embedder = w2v.Word2Vec(num_steps = 2500)
        self.costs = dict()

        if data != None:
            self.sym_dict = data.sym_dict
            self.offsets = data.offsets

            self.opcode_start = data.opcode_start
            self.operand_start = data.operand_start
            self.int_immed = data.int_immed
            self.float_immed = data.float_immed
            self.mem_start = data.mem_start

            self.costs = data.costs
            self.raw_data = data.raw_data

            self.final_embeddings = data.final_embeddings
            self.word2id = data.word2id
            self.fields = data.fields


    def set_embedding(self, embedding_file):

        """
        Optionally runs word2vec if an embedding file is not given or loads from file.

        """

        print 'getting the embedding...'
        if embedding_file == None:
            print 'running word2vec....'
            token_data = list()
            for row in self.raw_data:
                token_data.extend(row[0])
            print len(token_data)

            data = self.embedder.generate_dataset(token_data,self.sym_dict,self.mem_start)
            self.embedder.train(data,self.sym_dict,self.mem_start)
            self.final_embeddings = self.embedder.get_embedding()

            #create variable length per basic block instruction stream
            self.word2id = self.embedder.data.word2id
            self.id2word = self.embedder.data.id2word

        else:
            print 'reading from file....'
            with open(embedding_file,'r') as f:
                (self.final_embeddings,self.word2id,self.id2word) = torch.load(f)


    def extract_data(self, cnx, format, fields):
        print 'extracting data'
        self.raw_data = ut.get_data(cnx, format, fields)
        self.fields = fields

    def read_meta_data(self):

        self.sym_dict,_ = ut.get_sym_dict()
        self.offsets = ut.read_offsets()

        self.opcode_start = self.offsets[0]
        self.operand_start = self.offsets[1]
        self.int_immed = self.offsets[2]
        self.float_immed = self.offsets[3]
        self.mem_start = self.offsets[4]

        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = 1


    def generate_costdict(self, maxnum):
        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = np.random.randint(1,maxnum)

    def prepare_data(self):
        pass

    def generate_datasets(self):
        size = len(self.data)
        split = (size * self.percentage) // 100
        self.train  = self.data[:split]
        self.test = self.data[(split + 1):]
        print 'train ' + str(len(self.train)) + ' test ' + str(len(self.test))


    def generate_batch(self, batch_size, partition=None):
        if partition is None:
            partition = (0, len(self.train))

        # TODO: this seems like it would be expensive for a large data set
        (start, end) = partition
        population = range(start, end)
        selected = random.sample(population,batch_size)

        self.batch = []
        for index in selected:
            self.batch.append(self.train[index])

    def plot_histogram(self, data):

        ys = list()
        for item in data:
          ys.append(item.y)

        plt.hist(ys, min(max(ys), 1000))
        plt.show()
