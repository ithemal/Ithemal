#main data file

import numpy as np
import common_libs.utilities as ut
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


class Data(object):

    """
    Main data object which extracts data from a database, partition it and gives out batches.

    """


    def __init__(self): #copy constructor
        self.percentage = 80
        self.costs = dict()

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
