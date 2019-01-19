import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import common_libs.utilities as ut
import data.data_cost as dt
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple
from . import model_utils

class GraphNN(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_classes, use_residual=True, linear_embed=False, use_dag_rnn=True):
        # type: (int, int, int, bool, bool, bool) -> None

        super(GraphNN, self).__init__()

        self.num_classes = num_classes

        assert use_residual or use_dag_rnn, 'Must use some type of predictor'

        self.hidden_size = hidden_size
        self.use_residual = use_residual
        self.linear_embed = linear_embed
        self.use_dag_rnn = use_dag_rnn

        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size

        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

        # linear weight for instruction embedding
        self.opcode_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin = nn.Linear(self.embedding_size, self.hidden_size)
        # for sequential model
        self.opcode_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.src_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)
        self.dst_lin_seq = nn.Linear(self.embedding_size, self.hidden_size)

        #linear layer for final regression result
        self.linear = nn.Linear(self.hidden_size,self.num_classes)

        #lstm - for sequential model
        self.lstm_token_seq = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins_seq = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear_seq = nn.Linear(self.hidden_size, self.num_classes)

    def set_learnable_embedding(self, mode, dictsize, seed = None):
        # type: (str, int, Optional[int]) -> None

        self.mode = mode

        if mode != 'learnt':
            embedding = nn.Embedding(dictsize, self.embedding_size)

        if mode == 'none':
            print 'learn embeddings form scratch...'
            initrange = 0.5 / self.embedding_size
            embedding.weight.data.uniform_(-initrange, initrange)
            self.final_embeddings = embedding
        elif mode == 'seed':
            print 'seed by word2vec vectors....'
            embedding.weight.data = torch.FloatTensor(seed)
            self.final_embeddings = embedding
        elif mode == 'learnt':
            print 'using learnt word2vec embeddings...'
            self.final_embeddings = seed
        else:
            print 'embedding not selected...'
            exit()

    def dump_shared_params(self):
        # type: () -> Dict[str, Any]
        return model_utils.dump_shared_params(self)

    def load_shared_params(self, params):
    # type: (Dict[str, Any]) -> None
        model_utils.load_shared_params(self, params)

    def init_hidden(self):
        # type: () -> Tuple[autograd.Variable, autograd.Variable]

        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

    def remove_refs(self, item):
        # type: (dt.DataItem) -> None

       for instr in item.block.instrs:
            if instr.lstm != None:
                del instr.lstm
            if instr.hidden != None:
                del instr.hidden
            instr.lstm = None
            instr.hidden = None
            instr.tokens = None

    def init_bblstm(self, item):
        # type: (dt.DataItem) -> None

        self.remove_refs(item)
        for i, instr in enumerate(item.block.instrs):
            tokens = item.x[i]
            if self.mode == 'learnt':
                instr.tokens = [self.final_embeddings[token] for token in tokens]
            else:
                instr.tokens = self.final_embeddings(torch.LongTensor(tokens))

    def reduction(self, v1, v2):
        # type: (torch.tensor, torch.tensor) -> torch.tensor
        return torch.max(v1,v2)

    def create_graphlstm(self, block):
        # type: (ut.BasicBlock) -> torch.tensor

        roots = block.find_roots()

        root_hidden = []
        for root in roots:
            hidden = self.create_graphlstm_rec(root)
            root_hidden.append(hidden[0].squeeze())


        final_hidden = torch.zeros(self.hidden_size)
        if len(root_hidden) > 0:
            final_hidden = root_hidden[0]
        for hidden in root_hidden:
            final_hidden = self.reduction(final_hidden,hidden)

        return final_hidden

    def get_instruction_embedding(self, instr, seq_model):
        # type: (ut.Instruction, bool) -> torch.tensor

        if seq_model:
            opcode_lin = self.opcode_lin_seq
            src_lin = self.src_lin_seq
            dst_lin = self.dst_lin_seq
        else:
            opcode_lin = self.opcode_lin
            src_lin = self.src_lin
            dst_lin = self.dst_lin

        opc_embed = instr.tokens[0]
        src_embed = instr.tokens[2:2+len(instr.srcs)]
        dst_embed = instr.tokens[-1-len(instr.dsts):-1]

        opc_hidden = opcode_lin(opc_embed)

        src_hidden = torch.zeros(self.embedding_size)
        for s in src_embed:
            src_hidden = torch.max(F.relu(src_lin(s)))

        dst_hidden = torch.zeros(self.embedding_size)
        for d in dst_embed:
            dst_hidden = torch.max(F.relu(dst_lin(d)))

        return (opc_hidden + src_hidden + dst_hidden).unsqueeze(0).unsqueeze(0)

    def create_graphlstm_rec(self, instr):
        # type: (ut.Instruction) -> torch.tensor

        if instr.hidden != None:
            return instr.hidden

        parent_hidden = []
        for parent in instr.parents:
            hidden = self.create_graphlstm_rec(parent)
            parent_hidden.append(hidden)

        in_hidden_ins = self.init_hidden()
        if len(parent_hidden) > 0:
            in_hidden_ins = parent_hidden[0]
        h = in_hidden_ins[0]
        c = in_hidden_ins[1]
        for hidden in parent_hidden:
            h = self.reduction(h,hidden[0])
            c = self.reduction(c,hidden[1])
        in_hidden_ins = (h,c)

        ins_embed = self.get_instruction_embedding(instr, False)

        out_ins, hidden_ins = self.lstm_ins(ins_embed, in_hidden_ins)
        instr.hidden = hidden_ins

        return instr.hidden

    def create_residual_lstm(self, block):
        # type: (ut.BasicBlock) -> torch.tensor

        ins_embeds = autograd.Variable(torch.zeros(len(block.instrs),self.embedding_size))
        for i, ins in enumerate(block.instrs):
            ins_embeds[i] = self.get_instruction_embedding(ins, True)

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        _, hidden_ins = self.lstm_ins_seq(ins_embeds_lstm, self.init_hidden())

        seq_ret = hidden_ins[0].squeeze()

        return seq_ret


    def forward(self, item):
        # type: (dt.DataItem) -> torch.tensor

        self.init_bblstm(item)

        final_pred = torch.zeros(self.num_classes).squeeze()

        if self.use_dag_rnn:
            graph = self.create_graphlstm(item.block)
            final_pred += self.linear(graph).squeeze()

        if self.use_residual:
            sequential = self.create_residual_lstm(item.block)
            final_pred += self.linear_seq(sequential).squeeze()

        return final_pred.squeeze()
