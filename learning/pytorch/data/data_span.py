import sys
sys.path.append('..')
import numpy as np
import common_libs.utilities as ut
import random
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
from tqdm import tqdm

from data import Data
import random


class DataItem:

    def __init__(self, x, y, block):
        self.x = x
        self.y = y
        self.block = block


class DataSpan(Data):


    def __init__(self, data=None):
        super(DataSpan, self).__init__(data)

    def update_span(self, cnx, format):

        self.extract_data(cnx, format, ['code_id'])

        for row in tqdm(self.raw_data):
            sql = 'UPDATE code SET span=NULL WHERE code_id=' + str(row[1])
            ut.execute_query(cnx, sql, False)
        cnx.commit()

        blocks = []
        for row in self.raw_data:
            block = ut.create_basicblock(row[0])
            blocks.append(block)

        for i in tqdm(range(len(blocks))):
            span = blocks[i].num_span(self.costs)
            code_id = self.raw_data[i][1]
            sql = 'UPDATE code SET span=' + str(span) + ' WHERE code_id=' + str(code_id)
            ut.execute_query(cnx, sql, False)
        cnx.commit()


    def count_span_dist(self,data):
        spans = dict()
        for item in data:
            if item.y in spans:
                spans[item.y] += 1
            else:
                spans[item.y] = 1

        for key in sorted(spans.iterkeys()):
            sys.stdout.write("%d: %d, " % (key, spans[key]))
        sys.stdout.write('\n')

    def prepare_for_classification(self):

        for item in self.data:
            one_hot = [0] * self.max_span
            one_hot[item.y - 1] = 1
            item.y = one_hot

        return self.max_span


    def clean_data(self):

        #remove skew of data
        temp_data = []
        max_per_span = 20000
        spans = dict()

        for item in self.data:
            y = item.y
            if y in spans:
                spans[y] += 1
            else:
                spans[y] = 1

            if spans[y] <= max_per_span:
                temp_data.append(item)

        self.data = temp_data
        self.max_span = max(spans)

        print 'after removing skew...'
        print len(self.data)

        #randomize
        shuffled = range(len(self.data))
        random.shuffle(shuffled)

        temp_data = []
        for i in shuffled:
            temp_data.append(self.data[i])

        self.data = temp_data




class DataTokenEmbedding(DataSpan):

    def __init__(self, data=None):
        super(DataTokenEmbedding, self).__init__(data)

    def prepare_data(self):

        self.data = []

        span_wrong = 0
        span_right = 0

        print 'preparing data...'
        for row in tqdm(self.raw_data):
            if len(row[0]) > 0:
                code = []
                for token in row[0]:
                    code.append(self.word2id.get(token,0))

                block = ut.create_basicblock(row[0])
                span = block.num_span(self.costs)

                if span > 0:
                    span_right += 1
                else:
                    span_wrong += 1
                    continue

                block.create_dependencies()

                item = DataItem(code, span, block)
                self.data.append(item)

        print 'span wrong in ' + str(span_wrong)
        print 'data set size ' + str(span_right) + ' total database size ' +  str(len(self.raw_data))

        self.clean_data()



class DataInstructionEmbedding(DataSpan):

    def __init__(self,data=None):
        super(DataInstructionEmbedding, self).__init__(data)

    def prepare_data(self):

        self.data = []

        span_wrong = 0
        span_right = 0

        print 'preparing data...'
        for row in tqdm(self.raw_data):
            if len(row[0]) > 0:
                code = []
                ins = []
                for token in row[0]:
                    if token >= self.opcode_start and token < self.mem_start:
                        if len(ins) != 0:
                            code.append(ins)
                            ins = []
                    ins.append(self.word2id.get(token,0))

                if len(ins) != 0:
                    code.append(ins)
                    ins = []

                block = ut.create_basicblock(row[0])
                span = block.num_span(self.costs)

                if span > 0:
                    span_right += 1
                else:
                    span_wrong += 1
                    continue

                block.create_dependencies()

                item = DataItem(code, span, block)
                self.data.append(item)

        print 'span wrong in ' + str(span_wrong)
        print 'data set size ' + str(span_right) + ' total database size ' +  str(len(self.raw_data))

        self.clean_data()



