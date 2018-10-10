import numpy as np
import common_libs.utilities as ut
import random
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
from data import Data
from tqdm import tqdm

class DataItem:

    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost
        self.block = None


class DataTokenEmbedding(Data):

    """
    token embeddings (non-hierarchical) and
    num instructions calculated
    item.y - final cost
    item.cost - list of per instruction aggregate cost values

    """

    def __init__(self, data=None):
        super(DataTokenEmbedding, self).__init__(data)

    def prepare_data(self):

        self.data = []
        for row in tqdm(self.raw_data):
            if len(row[0]) > 0:
                count = 0
                labels = []
                code = []
                for token in row[0]:
                    code.append(self.word2id.get(token,0))
                    if token >= self.opcode_start and token < self.mem_start:
                        count += self.costs[token]
                    labels.append(count)
                item = DataItem(code, count, labels)
                self.data.append(item)


class DataInstructionEmbedding(Data):

    """
    token embeddings (hierarchical per instructions) and
    generated additive costs
    item.y - final cost
    item.cost - list of per instruction aggregate cost values

    """


    def __init__(self, data=None):
        super(DataInstructionEmbedding, self).__init__(data)

    def prepare_data(self):

        self.data = []

        for row in tqdm(self.raw_data):
            if len(row[0]) > 0:
                code = []
                ins = []
                cost = []
                count = 0
                for token in row[0]:
                    if token >= self.opcode_start and token < self.mem_start:
                        if len(ins) != 0:
                            code.append(ins)
                            ins = []
                        count += self.costs[token]
                        cost.append(count)
                    ins.append(self.word2id.get(token,0))
                if len(ins) != 0:
                    code.append(ins)
                    ins = []

                item = DataItem(code, count, cost)

                block = ut.create_basicblock(row[0])
                block.create_dependencies()
                item.block = block

                self.data.append(item)

