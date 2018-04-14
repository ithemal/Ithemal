from input_data import InputData
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import utilities as ut


# main interface class - give out all the parameters here
# this holds the final embeddings
# functions -
# __init__, train, get_embedding

class Word2Vec:
    def __init__(self,
                 #model related
                 emb_dimension=100,
                 
                 #data generation related
                 batch_size=50,
                 skip_window=2,
                 num_skips=2,
                 n_words=2000,
                 min_count=None,
                 word_dict=None, #if the inputs are already tokenized

                 #training related
                 epochs = 10,
                 initial_lr=0.025,
                 neg_words = 5):
       
        self.data = Data(num_skips=num_skips,
                         skip_windows=skip_window,
                         min_count=min_count,
                         n_words=n_words)
        
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size

        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

        self.epochs = epochs
        self.initial_lr = initial_lr
        self.neg_words = neg_words
       
        
    def generate_dataset(self, words):
        data = self.data.get_common_words(words)
        self.data.init_sample_table()
        return data

    def train(self, data):
        # will use epochal counting
        total_pair_count = len(data) * (2 * self.data.skip_window)
        batch_count = total_pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        for _ in range(epochs):
            for i in process_bar:
                pos_u, pos_v = self.data.generate_pos_pairs(self.batch_size)
                neg_v = self.data.generate_neg_words(self.batch_size, )
               
                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))
                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                self.optimizer.zero_grad()
                loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()

                process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data[0],
                                         self.optimizer.param_groups[0]['lr']))
                if i * self.batch_size % 10000 == 0:
                    lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

    def get_embedding(self):
        return self.skip_gram_model.get_embedding(self.use_cuda)
