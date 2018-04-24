#classifying if it's an opcode or not

#framework

#extract_and_prepare
#generate_datasets
#generate_batch
#generate_model
#train_model
#test_model
import numpy as np
import common.utilities as ut
import random  
import word2vec.word2vec as w2v
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch


class Data(object):
    
    def __init__(self):
        self.percentage = 80
        self.embedder = w2v.Word2Vec(num_steps = 2500)

    def extract_and_prepare_data(self,cnx,format,embedding_file):

        raw_data = ut.get_data(cnx,format,['num_instr'])
        if embedding_file == None:
            print 'running word2vec....'
            token_data = list()
            for row in raw_data:
                token_data.extend(row[0])
            print len(token_data)

            offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
            sym_dict, mem_start = ut.get_sym_dict(offsets_filename)
            offsets = ut.read_offsets(offsets_filename)

            data = self.embedder.generate_dataset(token_data,sym_dict,mem_start)
            self.embedder.train(data,sym_dict,mem_start)
            self.final_embeddings = self.embedder.get_embedding()
  
            #create variable length per basic block instruction stream
            word2id = self.embedder.data.word2id
        
        else:
            print 'reading from file....'
            with open(embedding_file,'r') as f:
                (self.final_embeddings, word2id,_) = torch.load(f)

        self.x = []
        self.y = []

        for row in raw_data:
            if row[1] != None and len(row[0]) > 0:
                code = []
                for token in row[0]:
                    code.append(self.final_embeddings[word2id.get(token,0)])
                self.x.append(code)
                self.y.append(row[1])

    def generate_datasets(self):
        assert len(self.x) == len(self.y)
        size = len(self.y)
        print len(self.x)
        split = (size * self.percentage) // 100
        self.train_x  = self.x[:split]
        self.train_y = self.y[:split]
        self.test_x = self.x[(split + 1):]
        self.test_y = self.y[(split + 1):]
        
    def generate_batch(self, batch_size):
        population = range(len(self.train_x))
        selected = random.sample(population,batch_size)
        batch_x = []
        batch_y = []
        for index in selected:
            batch_x.append(self.train_x[index])
            batch_y.append(self.train_y[index])

        return batch_x, batch_y


class Model(nn.Module):

    def __init__(self, embedding_size):
        super(Model, self).__init__()
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
        #num layers, minibatch size, hidden size
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        

    def forward(self, input):

        #print input
        #convert to tensor
        embeds = torch.FloatTensor(input)
        
        #prepare for lstm - seq len, batch size, embedding size
        seq_len = embeds.shape[0]
        embeds_for_lstm = embeds.view(seq_len, 1, -1)
        #print embeds_for_lstm.shape

        lstm_out, self.hidden = self.lstm(embeds_for_lstm, self.hidden)
        
        return self.linear(lstm_out[seq_len - 1,0,:]).squeeze()
  

class Train(): 


    def __init__(self, 
                 model,
                 data,
                 epochs = 5,
                 batch_size = 1000):
        self.model = model
        self.data = data
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001)

        #training parameters
        self.epochs = epochs
        self.batch_size = batch_size


    def train(self):

        epoch_len = (len(self.data.train_x) // self.batch_size) / 10
        print epoch_len

        for i in range(self.epochs):
           
            average_loss = 0
            j = 0
            
            for _ in range(epoch_len):
 
                batch_x, batch_y = self.data.generate_batch(self.batch_size)

                #we use batches of size one for actual training
                for x,y in zip(batch_x, batch_y):
                    
                    #zero out grads
                    self.optimizer.zero_grad()
                
                    #initialize hidden state
                    self.model.hidden = self.model.init_hidden()

                    #run the model
                    output = self.model(x)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output, torch.FloatTensor([y]))

                    average_loss = (average_loss * j + loss.item()) / (j + 1)
                    j += 1
                    
                    loss.backward()
                
                    #step the optimizer
                    self.optimizer.step()

                print j, average_loss
           
            print i


    def validate(self):
        
        for x,y in zip(self.data.test_x, self.data.test_y):
            
            self.model.hidden = self.model.init_hidden()
            output = self.model(x)
            print output.item(),y
            
            


