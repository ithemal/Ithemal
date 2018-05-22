import torch
import torch.nn as nn
import sys
sys.path.append('..')
import common.utilities as ut
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np

class InstructionLSTM(ut.Instruction):
    
    def __init__(self, opcode, srcs, dsts):
        super(InstructionLSTM, self).__init__(opcode, srcs, dsts)

    def init(self, tokens):
        self.lstm = None
        self.tokens = tokens
        self.hidden = None


class BasicBlockLSTM(ut.BasicBlock):

    def __init__(self, instrs):
        super(BasicBlocLSTM, self).__init__(instrs)


    def find_roots(self):
        roots = []
        for instr in self.instrs:
            if len(instr.children) == 0:
                roots.append(instr)

        return roots
        

class GraphNN(nn.Module):

    def __init__(self, embedding_size, num_classes):
        super(GraphNN, self).__init__()

        self.num_classes = num_classes

        self.hidden_size = 256
        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size
        
        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)
        
        #hidden state for the rnn
        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        #linear layer for final regression result
        self.linear = nn.Linear(self.hidden_size,self.num_classes)
        
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))


    def create_bblstm(self, x, block):

        block.__class__ = BasicBlockLSTM
        for i,instr in enumerate(block.instrs):
            instr.__class__ = InstructionLSTM
            instr.init(x[i])

    def reduction(self, v1, v2):
        return torch.max(v1,v2)

    def create_graphlstm(self, x, block):

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

    def create_graphlstm_rec(self, instr):

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

        #do the token based lstm encoding for the instruction
        token_embeds = torch.FloatTensor(instr.tokens)
        token_embeds_lstm = token_embeds.unsqueeze(1)
        out_token, hidden_token = self.lstm_token(token_embeds_lstm, self.hidden_token)

        ins_embed = hidden_token[0] #first out of the tuple
        
        out_ins, hidden_ins = self.lstm_ins(ins_embed,in_hidden_ins)
        instr.hidden = hidden_ins

        return instr.hidden

    def forward(self, item):

        
        self.hidden_ins = self.init_hidden()
        self.hidden_token = self.init_hidden()

        self.create_bblstm(item.x, item.block)

        hidden = self.create_graphlstm(item.x, item.block)
        
        return self.linear(hidden)


