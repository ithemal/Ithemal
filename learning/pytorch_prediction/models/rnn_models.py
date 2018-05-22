#this file contains models that I have tried out for different tasks, which are reusable
#plus it has the training framework for those models given data - each model has its own data requirements

import numpy as np
import common.utilities as ut
import random  
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch
import math


class ModelAbs(nn.Module):

    """
    Abstract model without the forward method.

    lstm for processing tokens in sequence and linear layer for output generation
    lstm is a uni-directional single layer lstm

    num_classes = 1 - for regression
    num_classes = n - for classifying into n classes

    """    

    def __init__(self, embedding_size, num_classes):

        super(ModelAbs, self).__init__()
        self.hidden_size = 256

        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        #lstm - input size, hidden size, num layers
        self.lstm_token = nn.LSTM(self.embedding_size, self.hidden_size)

        #hidden state for the rnn
        self.hidden_token = self.init_hidden()

        #linear layer for regression - in_features, out_features
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))


        
class ModelFinalHidden(nn.Module):

    """
    Prediction using the final hidden state of the unrolled rnn.

    Input - sequence of tokens processed in sequence by the lstm
    Output - the final value to be predicted

    we do not derive from ModelAbs, but instead use a bidirectional, multi layer 
    lstm and a deep MLP with non-linear activation functions to predict the final output
    
    """

    def __init__(self, embedding_size):
        super(ModelFinalHidden, self).__init__()
        self.hidden_size = 256
        self.embedding_size = embedding_size
        
        self.layers = 2
        self.directions = 1
        self.is_bidirectional = (self.directions == 2)
        self.lstm_token = torch.nn.LSTM(input_size = self.embedding_size, 
                                  hidden_size = self.hidden_size,
                                  num_layers = self.layers,
                                  bidirectional = self.is_bidirectional)
        self.linear1 = nn.Linear(self.layers * self. directions * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,1)
        self.hidden_token = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.layers * self.directions, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(self.layers * self.directions, 1, self.hidden_size)))

    def forward(self, item):

        self.hidden_token = self.init_hidden()

        #convert to tensor
        embeds = torch.FloatTensor(item.x)
        
        #prepare for lstm - seq len, batch size, embedding size
        seq_len = embeds.shape[0]
        embeds_for_lstm = embeds.unsqueeze(1)

        lstm_out, self.hidden_token = self.lstm_token(embeds_for_lstm, self.hidden_token)
        
        f1 = nn.functional.relu(self.linear1(self.hidden_token[0].squeeze().view(-1)))
        f2 = self.linear2(f1)
        return f2
    
class ModelAggregate(ModelAbs):

    """
    Prediction at every hidden state of the unrolled rnn.

    Input - sequence of tokens processed in sequence by the lstm
    Output - predictions at the every hidden state

    uses lstm and linear setup of ModelAbs
    each hidden state is given as a seperate batch to the linear layer
    
    """

    def __init__(self, embedding_size):
        super(ModelAggregate, self).__init__(embedding_size,1)
   
    def forward(self, item):

        self.hidden_token = self.init_hidden()
 
        #convert to tensor
        embeds = torch.FloatTensor(item.x)
        
        #prepare for lstm - seq len, batch size, embedding size
        seq_len = embeds.shape[0]
        embeds_for_lstm = embeds.unsqueeze(1)

        lstm_out, self.hidden_token = self.lstm_token(embeds_for_lstm, self.hidden_token)
    
        values = self.linear(lstm_out[:,0,:].squeeze()).squeeze()
        
        return values

class ModelInstructionAggregate(ModelAbs):

    """
    Prediction at every hidden state of the unrolled rnn for instructions.

    Input - sequence of tokens processed in sequence by the lstm but seperated into instructions
    Output - predictions at the every hidden state

    lstm predicting instruction embedding for sequence of tokens
    lstm_ins processes sequence of instruction embeddings
    linear layer process hidden states to produce output
    
    """
    
    def __init__(self, embedding_size):
        super(ModelInstructionAggregate, self).__init__(embedding_size,1)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

    def copy(self, model):

        self.linear = model.linear
        self.lstm_token = model.lstm_token
        self.lstm_ins = model.lstm_ins

    def forward(self, item):
        
        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.embedding_size))
        for i, ins in enumerate(item.x):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        return self.linear(out_ins[:,0,:]).squeeze(1)


class ModelInstructionEmbedding(ModelAbs):

    """
    Prediction at the final hidden state of the unrolled rnn for instructions.

    Input - sequence of tokens processed in sequence by the lstm but seperated into instructions
    Output - predictions at the every hidden state

    lstm predicting instruction embedding for sequence of tokens
    lstm_ins processes sequence of instruction embeddings
    linear layer process hidden states to produce output
    
    """

    
    def __init__(self, embedding_size):
        super(ModelInstructionEmbedding, self).__init__(embedding_size, 1)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, item):

        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        
        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.embedding_size))
        for i, ins in enumerate(item.x):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        return self.linear(hidden_ins[0].squeeze())

class ModelInstructionEmbeddingClassification(ModelAbs):
    
    """
    Prediction at the final hidden state of the unrolled rnn for instructions.
    Selected from a given set of outputs - classification
    softmax with cross entropy as the loss

    """
    
    def __init__(self, embedding_size, num_classes):
        super(ModelInstructionEmbeddingClassification, self).__init__(embedding_size, num_classes)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)
        

    def forward(self, item):
        
        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.embedding_size))
        for i, ins in enumerate(item.x):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        return self.linear(hidden_ins[0].squeeze()).squeeze()

class ModelSpanRelational(ModelAbs):
    
    def __init__(self, embedding_size):
        super(ModelSpanRelational, self).__init__(embedding_size, 1)
        
        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)
        
        self.linearg1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linearg2 = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, item):

        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.hidden_size))
        for i, ins in enumerate(item.x):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        seq_len = len(item.x)
        
        g_variable = autograd.Variable(torch.zeros(self.hidden_size))

        for i in range(seq_len):
            for j in range(i,seq_len):
                
                concat = torch.cat((out_ins[i].squeeze(),out_ins[j].squeeze()),0)
                g1 = nn.functional.relu(self.linearg1(concat))
                g2 = nn.functional.relu(self.linearg2(g1))

                g_variable += g2


        output = self.linear(g_variable)

        return output

class ModelSpanRelationalClassification(ModelAbs):
    
    def __init__(self, embedding_size,num_classes):
        super(ModelSpanRelationalClassification, self).__init__(embedding_size, num_classes)
        
        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)
        
        self.linearg1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linearg2 = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, item):

        self.hidden_token = self.init_hidden()
        self.hidden_ins = self.init_hidden()

        ins_embeds = autograd.Variable(torch.zeros(len(item.x),self.hidden_size))
        for i, ins in enumerate(item.x):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_token(token_embeds_lstm,self.hidden_token)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm_ins(ins_embeds_lstm, self.hidden_ins)

        seq_len = len(item.x)
        
        g_variable = autograd.Variable(torch.zeros(self.hidden_size))

        for i in range(seq_len):
            for j in range(i,seq_len):
                
                concat = torch.cat((out_ins[i].squeeze(),out_ins[j].squeeze()),0)
                g1 = nn.functional.relu(self.linearg1(concat))
                g2 = nn.functional.relu(self.linearg2(g1))

                g_variable += g2


        output = self.linear(g_variable)

        return output
        
