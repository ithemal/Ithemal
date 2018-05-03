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
        self.costs = dict()
    
    def extract_data(self,cnx,format,embedding_file):

        self.raw_data = ut.get_data(cnx,format,['num_instr'])
        offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
        self.sym_dict,_ = ut.get_sym_dict(offsets_filename)
        self.offsets = ut.read_offsets(offsets_filename)

        print self.offsets
        self.opcode_start = self.offsets[0]
        self.operand_start = self.offsets[1]
        self.int_immed = self.offsets[2]
        self.float_immed = self.offsets[3]
        self.mem_start = self.offsets[4]

        if embedding_file == None:
            print 'running word2vec....'
            token_data = list()
            for row in self.raw_data:
                token_data.extend(row[0])
            print len(token_data)
            
            data = self.embedder.generate_dataset(token_data,self.sym_dict,self.mem_start)
            self.embedder.train(data,self.sym_dict,self.mem_start)
            self.final_embeddings = self.embedder.get_embedding()
  
            #create variable length per basic block instruction stream
            self.word2id = self.embedder.data.word2id
        
        else:
            print 'reading from file....'
            with open(embedding_file,'r') as f:
                (self.final_embeddings,self.word2id,_) = torch.load(f)

        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = 1

    def prepare_data(self):
        Pass

    def generate_costdict(self):
        for i in range(self.opcode_start, self.mem_start):
            self.costs[i] = np.random.randint(1,20)

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


class DataFinalHidden(Data):

    def __init__(self):
        super(DataFinalHidden, self).__init__()
    
    def prepare_data(self):
        self.x = []
        self.y = []
    
        for row in self.raw_data:
            if row[1] != None and len(row[0]) > 0:
                code = []
                for token in row[0]:
                    code.append(self.final_embeddings[self.word2id.get(token,0)])
                self.x.append(code)
                self.y.append(row[1])

class DataAggregate(Data):


    def __init__(self):
        super(DataAggregate, self).__init__()
    
    def prepare_data(self):
    
        self.x = []
        self.y = []
    
        for row in self.raw_data:
            if row[1] != None and len(row[0]) > 0:
                code = []
                labels = []
                count = 0
                for token in row[0]:
                    code.append(self.final_embeddings[self.word2id.get(token,0)])
                    if token >= self.opcode_start and token < self.mem_start:
                        count += self.costs[token]
                    labels.append(count)
                self.x.append(code)
                self.y.append(labels)

class DataInstructionEmbedding(Data):
    
    def __init__(self):
        super(DataInstructionEmbedding, self).__init__()

    def prepare_data(self):
        self.x = []
        self.y = []

        wrong = 0
    
        for row in self.raw_data:
            if row[1] != None and len(row[0]) > 0:
                code = []
                ins = []
                count = 0
                for token in row[0]:
                    if token >= self.opcode_start and token < self.mem_start:
                        if len(ins) != 0:
                            code.append(ins)
                            ins = []
                        count += self.costs[token]
                    ins.append(self.final_embeddings[self.word2id.get(token,0)]) 
                if len(ins) != 0:
                    code.append(ins)
                    ins = []
                if len(code) != row[1]:
                    wrong += 1 
                #assert len(code) == row[1]
                if len(code) == row[1]:
                    self.x.append(code)
                    self.y.append(count)
        
        print wrong
        print len(self.x)
        

class ModelAbs(nn.Module):
    
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

class ModelInstructionEmbedding(ModelAbs):
    
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
        

class Train(): 


    def __init__(self, 
                 model,
                 data,
                 epochs = 10,
                 batch_size = 1000):
        self.model = model
        print self.model
        self.data = data
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

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
                    outputs = self.model(x)
                    targets = torch.FloatTensor([y]).squeeze()
                    
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()


                    average_loss = (average_loss * j + loss.item()) / (j + 1)
                    j += 1
                    
                    
                    #step the optimizer
                
                print j, average_loss
           
            print i


    def validate(self):
        
        average_loss = 0
        j = 0

        f = open('output.txt','w')

        print len(self.data.test_x)

        for x,y in zip(self.data.test_x, self.data.test_y):
            
            self.model.hidden = self.model.init_hidden()
            output = self.model(x)
            output_list = output.data.numpy().tolist()
            
            if not isinstance(y, list):
                f.write('%f,%d ' % (output.data[0],y))
            else:
                if len(y) == 1:
                    f.write('%f,%d ' % (output_list,y[0]))
                else:
                    for i,_ in enumerate(y):
                        f.write('%f,%d ' % (output_list[i],y[i]))
            f.write('\n')

            targets = torch.FloatTensor([y]).squeeze()
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, targets)

            average_loss = (average_loss * j + loss.item())/(j+1)
            j += 1
            if j % 1000 == 0:
                print j, average_loss

        print average_loss
        f.close()
            
            



            
            


