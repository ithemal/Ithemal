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

    lstm for processing input in sequence and linear layer for regression
    lstm is a uni-directional single layer lstm

    """    

    def __init__(self, embedding_size):
        super(ModelAbs, self).__init__()
        self.hidden_size = 256
        #numpy array with batchsize, embedding_size
        self.embedding_size = embedding_size
        
        #lstm - input size, hidden size, num layers
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size)

        #hidden state for the rnn
        self.hidden = self.init_hidden()

        #linear layer for regression - in_features, out_features
        self.linear = nn.Linear(self.hidden_size, 1)
    
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
        self.lstm = torch.nn.LSTM(input_size = self.embedding_size, 
                                  hidden_size = self.hidden_size,
                                  num_layers = self.layers,
                                  bidirectional = self.is_bidirectional)
        self.linear1 = nn.Linear(self.layers * self. directions * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.layers * self.directions, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(self.layers * self.directions, 1, self.hidden_size)))

    def forward(self, input):

        #print input
        #convert to tensor
        embeds = torch.FloatTensor(input)
        #print embeds.shape
        
        #prepare for lstm - seq len, batch size, embedding size
        seq_len = embeds.shape[0]
        #embeds_for_lstm = embeds.view(seq_len, 1, -1)
        embeds_for_lstm = embeds.unsqueeze(1)
        #print embeds_for_lstm.shape

        lstm_out, self.hidden = self.lstm(embeds_for_lstm, self.hidden)
        
        f1 = nn.functional.relu(self.linear1(self.hidden[0].squeeze().view(-1)))
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
        super(ModelAggregate, self).__init__(embedding_size)
   
    def forward(self, input):

        #print input
        #convert to tensor
        embeds = torch.FloatTensor(input)
        #print embeds.shape
        
        #prepare for lstm - seq len, batch size, embedding size
        seq_len = embeds.shape[0]
        #embeds_for_lstm = embeds.view(seq_len, 1, -1)
        embeds_for_lstm = embeds.unsqueeze(1)
        #print embeds_for_lstm.shape

        lstm_out, self.hidden = self.lstm(embeds_for_lstm, self.hidden)
    
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
        super(ModelInstructionAggregate, self).__init__(embedding_size)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.embedding_size, self.hidden_size)

    def copy(self, model):

        self.lstm = model.lstm
        self.linear = model.linear
        self.lstm_ins = model.lstm_ins

    def forward(self, inputs):
        
        ins_embeds = autograd.Variable(torch.zeros(len(inputs),self.embedding_size))
        for i, ins in enumerate(inputs):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_ins(token_embeds_lstm,self.hidden_ins)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm(ins_embeds_lstm, self.hidden)

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
        super(ModelInstructionEmbedding, self).__init__(embedding_size)

        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.embedding_size, self.hidden_size)

    def forward(self, inputs):
        
        ins_embeds = autograd.Variable(torch.zeros(len(inputs),self.embedding_size))
        for i, ins in enumerate(inputs):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_ins(token_embeds_lstm,self.hidden_ins)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm(ins_embeds_lstm, self.hidden)

        return self.linear(hidden_ins[0].squeeze())
        

class ModelSpanRelational(ModelAbs):
    
    def __init__(self, embedding_size):
        super(ModelSpanRelational, self).__init__(embedding_size)
        
        self.hidden_ins = self.init_hidden()
        self.lstm_ins = nn.LSTM(self.hidden_size, self.hidden_size)
        
        self.linearg1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linearg2 = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, inputs):

        ins_embeds = autograd.Variable(torch.zeros(len(inputs),self.hidden_size))
        for i, ins in enumerate(inputs):
            token_embeds = torch.FloatTensor(ins)
            token_embeds_lstm = token_embeds.unsqueeze(1)
            out_token, hidden_token = self.lstm_ins(token_embeds_lstm,self.hidden_ins)
            ins_embeds[i] = hidden_token[0].squeeze()

        ins_embeds_lstm = ins_embeds.unsqueeze(1)

        out_ins, hidden_ins = self.lstm(ins_embeds_lstm, self.hidden)

        seq_len = len(inputs)
        
        g_variable = autograd.Variable(torch.zeros(self.hidden_size))

        for i in range(seq_len):
            for j in range(i,seq_len):
                
                concat = torch.cat((out_ins[i].squeeze(),out_ins[j].squeeze()),0)
                g1 = nn.functional.relu(self.linearg1(concat))
                g2 = nn.functional.relu(self.linearg2(g1))

                g_variable += g2


        output = self.linear(g_variable)

        return output
        

class Train(): 

    """
    Performs training and validation for the models listed above

    """

    def __init__(self, 
                 model,
                 data,
                 epochs = 3,
                 batch_size = 1000):
        self.model = model
        print self.model
        self.data = data
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)

        #training parameters
        self.epochs = epochs
        self.batch_size = batch_size

    """
    MSELoss.
    """

    def mse_loss(self,x,y,printval):
        outputs = self.model(x)
        targets = torch.FloatTensor([y]).squeeze()
        
        if printval:
            print outputs.data.numpy()
            print targets.data.numpy()

        loss_fn = nn.MSELoss()
        loss = torch.sqrt(loss_fn(outputs, targets))

        if math.isnan(loss.item()):
            print outputs
            print targets
            print y
            exit()
        return [loss]
       
    """
    MSELoss + margin rank loss
    """

    def mse_loss_plus_rank_loss(self,x,y,printval):
        outputs = self.model(x)
        
        cost = outputs[-1]
        target_cost = torch.FloatTensor([y]).squeeze()

        if printval:
            print cost.data.numpy()
            print target_cost.data.numpy()

        
        if outputs.size()[0] > 1:
            inter = outputs[:-1]
            inter_1 = outputs[1:]
        else: #emulate no rank loss
            inter = torch.ones(1)
            inter_1 = 2 * torch.ones(1)
                        
        target_rank = torch.ones(inter.size())
        
        loss_mse = nn.MSELoss()
        loss1 = torch.sqrt(loss_mse(cost, target_cost)) / y
        
        loss_rank = nn.MarginRankingLoss()
        loss2 = loss_rank(inter_1, inter, target_rank)
        
        return [loss1, loss2]
                    
    """
    Training loop - to do make the average loss for general
    """

    def train(self, loss_fn, num_losses):

        epoch_len = (len(self.data.train_x) // self.batch_size) / 10

        print 'start training...'
        print 'epochs ' + str(self.epochs)
        print 'epoch length ' + str(epoch_len)
        print 'batch size (sampled) ' + str(self.batch_size)

        for i in range(self.epochs):
           
            average_loss = [0] * num_losses
            j = 0
            
            for _ in range(epoch_len):
 
                batch_x, batch_y = self.data.generate_batch(self.batch_size)

                average_loss_per_batch = [0] * num_losses
                batch_j = 0

                #we use batches of size one for actual training
                for x,y in zip(batch_x, batch_y):
                    
                    #zero out grads
                    self.optimizer.zero_grad()
                
                    #initialize hidden state
                    self.model.hidden = self.model.init_hidden()

                    #initialize the hidden state for instructions - this should be an aggregate model
                    if isinstance(self.model, ModelInstructionAggregate) or isinstance(self.model, ModelSpanRelational) or isinstance(self.model, ModelInstructionEmbedding):
                        self.model.hidden_ins = self.model.init_hidden()

                    #run the model
                    losses = loss_fn(x,y, batch_j == (self.batch_size / 2))

                    loss = torch.zeros(1)
                    for c,l in enumerate(losses):
                        loss += l
                        average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)
                        average_loss_per_batch[c] = (average_loss_per_batch[c] * batch_j  + l.item()) / (batch_j + 1)

                    loss.backward()
                    self.optimizer.step()

                    j += 1
                    batch_j += 1
                                        
                    #step the optimizer
                
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                for av in average_loss_per_batch:
                    p_str += str(av) + ' '
                print p_str
                
           
            print i
        
    """
    Validation with a test set
    """

    def validate(self, loss_fn, num_losses):
        
        average_loss = [0] * num_losses
        j = 0

        f = open('output.txt','w')

        print len(self.data.test_x)

        for x,y in zip(self.data.test_x, self.data.test_y):
            
            self.model.hidden = self.model.init_hidden()
            
            if isinstance(self.model, ModelInstructionAggregate):
                self.model.hidden_ins = self.model.init_hidden()

            outputs = self.model(x)

            output_list = outputs.data.numpy().tolist()[-1]
            
            if not isinstance(y, list):
                f.write('%f,%d ' % (outputs.data[0],y))
            else:
                if len(y) == 1:
                    f.write('%f,%d ' % (output_list,y[0]))
                else:
                    for i,_ in enumerate(y):
                        f.write('%f,%d ' % (output_list[i],y[i]))
            f.write('\n')

            losses = loss_fn(x,y, j % self.batch_size == 0)

            loss = torch.zeros(1)
            for c,l in enumerate(losses):
                loss += l
                average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)
            
            j += 1
            if j % 1000 == 0:
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                print p_str
           
        print average_loss
        f.close()            



            
            


