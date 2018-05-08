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
from tqdm import tqdm


class Data(object):
    
    def __init__(self):
        self.percentage = 80
        self.embedder = w2v.Word2Vec(num_steps = 2500)
        self.costs = dict()
    
    def extract_data(self,cnx,format,embedding_file):

        self.raw_data = ut.get_data(cnx,format,['code_id','time'])
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

 
    def prepare_data(self):
        Pass

    def generate_datasets(self):
        assert len(self.x) == len(self.y)
        size = len(self.y)
        print len(self.x)
        split = (size * self.percentage) // 100
        print split
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


class DataInstructionEmbedding(Data):
    
    def __init__(self):
        super(DataInstructionEmbedding, self).__init__()
        self.time_percentage = 10
        self.threshold = 80

    def get_time_mode(self, times):
        self.times = []
        for row in times:
            if row[0] < 1 or row[0] > 100000: #something is wrong for this code so just omit??
                break
            inserted = False
            try:
                for i,time in enumerate(self.times):
                    if (abs(row[0] - time[0]) * 100) / time[0] < self.time_percentage:
                        new_time = (time[0] * time[1] + row[0] * row[1]) / (time[1] + row[1])
                        new_count = time[1] + row[1]
                        self.times[i] = [new_time, new_count]
                        inserted = True
                        break
                if not inserted:
                    self.times.append([row[0],row[1]])
            except:
                print self.times

        maxcount = 0
        maxtime = 0
        total = 0
        for time in self.times:
            total += time[1]
            if time[1] >= maxcount:
                maxcount = time[1]
                maxtime = time[0]

        if total > 0 and maxcount * 100 / total >= self.threshold:
            return maxtime
        else:
            return None

    def update_times(self, cnx):

        count = 0
        for row in tqdm(self.raw_data):
            sql = 'UPDATE code SET time=NULL WHERE code_id=' + str(row[1])
            ut.execute_query(cnx, sql, False)
        cnx.commit()

        for row in tqdm(self.raw_data):
            sql = 'SELECT time, count FROM times WHERE code_id = ' + str(row[1])
            times = ut.execute_query(cnx, sql, True)                
            mode = self.get_time_mode(times)
            if mode != None:
                count += 1
                sql = 'UPDATE code SET time=' + str(mode) + ' WHERE code_id=' + str(row[1])
                ut.execute_query(cnx, sql, False)
                sql = 'SELECT time from code WHERE code_id=' + str(row[1])
                res = ut.execute_query(cnx, sql, True)
                assert res[0][0] == mode

        print len(self.raw_data)
        print count

        cnx.commit()
                
                
        
        
    def prepare_data(self, cnx):
        self.x = []
        self.y = []

        for row in tqdm(self.raw_data):
            if len(row[0]) > 0 and row[2] != None:
            
                code = []
                ins = []
                for token in row[0]:
                    if token >= self.opcode_start and token < self.mem_start:
                        if len(ins) != 0:
                            code.append(ins)
                            ins = []
                    ins.append(self.final_embeddings[self.word2id.get(token,0)]) 
                if len(ins) != 0:
                    code.append(ins)
                    ins = []
                mode = row[2]
      
                self.x.append(code)
                self.y.append(mode)
        
        print len(self.raw_data)
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
                 epochs = 7,
                 batch_size = 1000):
        self.model = model
        print self.model
        self.data = data
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

        #training parameters
        self.epochs = epochs
        self.batch_size = batch_size


    def train(self):

        print len(self.data.train_x), len(self.data.test_x)
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

                    #initialize the hidden state for instructions
                    if isinstance(self.model, ModelInstructionEmbedding): 
                        self.model.hidden_ins = self.model.init_hidden()

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
            
            if isinstance(self.model, ModelInstructionEmbedding):
                self.model.hidden_ins = self.model.init_hidden()

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
            
            



            
            


