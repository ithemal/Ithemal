import torch
import torch.nn as nn
import sys
sys.path.append('..')
import common_libs.utilities as ut
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np


class GraphNN(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_classes, use_residual=True):
        super(GraphNN, self).__init__()

        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.use_residual = use_residual

        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size

        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

        #hidden state for the rnn
        #self.hidden_token = self.init_hidden()
        #self.hidden_ins = self.init_hidden()

        #linear layer for final regression result
        self.linear = nn.Linear(self.hidden_size,self.num_classes)

        #lstm - for sequential model
        self.lstm_token_seq = nn.LSTM(self.embedding_size, self.hidden_size)
        self.lstm_ins_seq = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear_seq = nn.Linear(self.hidden_size, self.num_classes)

    def set_learnable_embedding(self, mode, dictsize, seed = None):

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


    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

    def remove_refs(self, item):

       for instr in item.block.instrs:
            if instr.lstm != None:
                del instr.lstm
            if instr.hidden != None:
                del instr.hidden
            instr.lstm = None
            instr.hidden = None
            instr.tokens = None

    def init_bblstm(self, item):

        self.remove_refs(item)
        for i, instr in enumerate(item.block.instrs):
            tokens = item.x[i]
            if self.mode == 'learnt':
                instr.tokens = []
                for token in tokens:
                    instr.tokens.append(self.final_embeddings[token])
            else:
                instr.tokens = self.final_embeddings(torch.LongTensor(tokens))

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
        in_hidden_token = self.init_hidden()
        out_token, hidden_token = self.lstm_token(token_embeds_lstm,in_hidden_token)

        ins_embed = hidden_token[0] #first out of the tuple

        out_ins, hidden_ins = self.lstm_ins(ins_embed,in_hidden_ins)
        instr.hidden = hidden_ins

        return instr.hidden

    def create_residual_lstm(self, x, block):

        hidden_token  = self.init_hidden()
        hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(x),self.embedding_size))
        for i, ins in enumerate(x):

            token_embeds = block.instrs[i].tokens

            #token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token_seq(token_embeds_lstm,hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins_seq(ins_embeds_lstm, hidden_ins)

        seq_ret = hidden_ins[0].squeeze()

        return seq_ret


    def forward(self, item):


        #self.hidden_ins = self.init_hidden()
        #self.hidden_token = self.init_hidden()

        self.init_bblstm(item)

        graph = self.create_graphlstm(item.x, item.block)

        if self.use_residual:
            sequential = self.create_residual_lstm(item.x, item.block)
            final = self.linear(graph).squeeze() + self.linear_seq(sequential).squeeze()
        else:
            final = self.linear(graph).squeeze()

        #final = self.reduction(graph, sequential)
        #return self.linear(final).squeeze()

        #print final.item()

        return final
