#classifying if it's an opcode or not

#framework

#extract_and_prepare
#generate_datasets
#generate_batch
#generate_model
#train_model
#test_model
import numpy as np
import utilities as ut
import random  
import word2vec.word2vec as w2v


class Data(object):
    
    def __init__(self):
        self.percentage = 80
        self.embedder = w2v.Word2Vec()

    def extract_and_prepare_data(self,cnx,format):

        raw_data = ut.get_data(cnx,format,[])
        token_data = list()
        for row in raw_data:
            token_data.extend(row[0])
        print len(token_data)

        offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
        sym_dict, mem_start = ut.get_sym_dict(offsets_filename)
        offsets = ut.read_offsets(offsets_filename)

        data = self.embedder.generate_datasets(token_data)
        self.embedder.train(data)
        final_embeddings = self.embedder.get_embedding()
        
        embedding_size = self.embedder.emb_dimension
    
        #create the entire dataset from the learnt embeddings
        self.x = np.ndarray(shape = [len(dict_data),embedding_size]) 
        self.y = np.ndarray(shape = [len(dict_data),1])

        for i,token in enumerate(token_data):        
            self.x[i] = final_embeddings[dict_data[i]]
            if token >= offsets[0] and token < offsets[4]:
                self.y[i] = 1
            else:
                self.y[i] = 0

    def generate_datasets(self):
        assert self.x.shape[0] == self.y.shape[0]
        size = self.y.shape[0]
        split = (size * self.percentage) // 100
        self.train_x  = self.x[:split,:]
        self.train_y = self.y[:split]
        self.test_x = self.x[(split + 1):,:]
        self.test_y = self.y[(split + 1):]
        
    def generate_batch(self, batch_size):
        population = range(self.train_x.shape[0])
        embedding_size = self.embedder.embedding_size
        selected = random.sample(population,batch_size)
        batch_x = np.ndarray(shape = [batch_size, embedding_size])
        batch_y = np.ndarray(shape = [batch_size,1])
        for i,index in enumerate(selected):
            batch_x[i] = self.train_x[index,:]
            batch_y[i] = self.train_y[index]

        return batch_x, batch_y
  
 




