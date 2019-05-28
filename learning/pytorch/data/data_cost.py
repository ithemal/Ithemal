import numpy as np
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
from tqdm import tqdm
from .data import Data
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import xml.etree.ElementTree as ET
import itertools

import sys
sys.path.append('..')

import common_libs.utilities as ut


class DataItem:

    def __init__(self, x, y, block, code_id):
        self.x = x
        self.y = y
        self.block = block
        self.code_id = code_id

class DataInstructionEmbedding(Data):

    def __init__(self):
        super(DataInstructionEmbedding, self).__init__()
        self.token_to_hot_idx = {}
        self.hot_idx_to_token = {}
        self.data = []

    def dump_dataset_params(self):
        return (self.token_to_hot_idx, self.hot_idx_to_token)

    def load_dataset_params(self, params):
        (self.token_to_hot_idx, self.hot_idx_to_token) = params

    def prepare_data(self, progress=True, fixed=False):
        def hot_idxify(elem):
            if elem not in self.token_to_hot_idx:
                if fixed:
                    # TODO: this would be a good place to implement UNK tokens
                    raise ValueError('Ithemal does not yet support UNK tokens!')
                self.token_to_hot_idx[elem] = len(self.token_to_hot_idx)
                self.hot_idx_to_token[self.token_to_hot_idx[elem]] = elem
            return self.token_to_hot_idx[elem]

        if progress:
            iterator = tqdm(self.raw_data)
        else:
            iterator = self.raw_data

        for (code_id, timing, code_intel, code_xml) in iterator:
            block_root = ET.fromstring(code_xml)
            instrs = []
            raw_instrs = []
            curr_mem = self.mem_start
            for _ in range(1): # repeat for duplicated blocks
                # handle missing or incomplete code_intel
                split_code_intel = itertools.chain((code_intel or '').split('\n'), itertools.repeat(''))
                for (instr, m_code_intel) in zip(block_root, split_code_intel):
                    raw_instr = []
                    opcode = int(instr.find('opcode').text)
                    raw_instr.extend([opcode, '<SRCS>'])
                    srcs = []
                    for src in instr.find('srcs'):
                        if src.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in src.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            srcs.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(src.text))
                            srcs.append(int(src.text))

                    raw_instr.append('<DSTS>')
                    dsts = []
                    for dst in instr.find('dsts'):
                        if dst.find('mem') is not None:
                            raw_instr.append('<MEM>')
                            for mem_op in dst.find('mem'):
                                raw_instr.append(int(mem_op.text))
                                # operands used to calculate dst mem ops are sources
                                srcs.append(int(mem_op.text))
                            raw_instr.append('</MEM>')
                            dsts.append(curr_mem)
                            curr_mem += 1
                        else:
                            raw_instr.append(int(dst.text))
                            dsts.append(int(dst.text))

                    raw_instr.append('<END>')
                    raw_instrs.append(list(map(hot_idxify, raw_instr)))
                    instrs.append(ut.Instruction(opcode, srcs, dsts, len(instrs)))
                    instrs[-1].intel = m_code_intel

            block = ut.BasicBlock(instrs)
            block.create_dependencies()
            datum = DataItem(raw_instrs, timing, block, code_id)
            self.data.append(datum)

def load_dataset(data_savefile=None, arch=None, format='text'):
    data = DataInstructionEmbedding()

    if data_savefile is None:
        if arch is None:
            raise ValueError('Must provide one of data_savefile or arch')

        cnx = ut.create_connection()
        data.extract_data(cnx, format, ['code_id','code_intel'])
        data.get_timing_data(cnx, arch)
    else:
        data.raw_data = torch.load(data_savefile)

    data.read_meta_data()
    data.prepare_data()
    data.generate_datasets()

    return data
