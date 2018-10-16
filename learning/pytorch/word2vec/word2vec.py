from input_data import Data
import numpy as np
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
from sklearn.metrics.pairwise import cosine_similarity
import common_libs.utilities as ut
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



# main interface class - give out all the parameters here
# this holds the final embeddings
# functions -
# __init__, train, get_embedding

class Word2Vec:
    def __init__(self,
                 #model related
                 emb_dimension=256,

                 #data generation related
                 batch_size=1024,
                 skip_window=2,
                 num_skips=2,
                 n_words=2000,
                 min_count=None,
                 word_dict=None, #if the inputs are already tokenized

                 #training related
                 epochs = 5,
                 initial_lr=0.001,
                 num_steps = None,
                 neg_words = 32):

        self.data = Data(num_skips=num_skips,
                         skip_window=skip_window,
                         min_count=min_count,
                         n_words=n_words)

        self.emb_dimension = emb_dimension
        self.batch_size = batch_size

        self.epochs = epochs
        self.initial_lr = initial_lr
        self.neg_words = neg_words
        self.num_steps = num_steps


    def generate_dataset(self, words, sym_dict, mem_offset):
        data = self.data.get_common_words(words, sym_dict, mem_offset)
        self.data.init_sample_table()
        self.emb_size = self.data.word_count
        return data

    def train(self, data, sym_dict, mem_offset):

        try:
            print self.emb_size, self.emb_dimension
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
            self.use_cuda = torch.cuda.is_available()
            #self.use_cuda = False

            if self.use_cuda:
                self.skip_gram_model.cuda()
            self.optimizer = optim.Adam(
                self.skip_gram_model.parameters(), lr=self.initial_lr)

            # will use epochal counting
            total_pair_count = len(data) * (2 * self.data.skip_window)
            if self.num_steps == None:
                batch_count = total_pair_count / self.batch_size
            else:
                batch_count = self.num_steps / self.epochs
            # self.skip_gram_model.save_embedding(
            #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
            for j in range(self.epochs):
                process_bar = tqdm(range(int(batch_count)))
                for i in process_bar:
                    pos_u, pos_v = self.data.generate_pos_pairs(data, self.batch_size)
                    neg_v = self.data.generate_neg_words(self.batch_size, self.neg_words)

                    #print pos_u, pos_v, neg_v

                    pos_u = Variable(torch.LongTensor(pos_u))
                    pos_v = Variable(torch.LongTensor(pos_v))
                    neg_v = Variable(torch.LongTensor(neg_v))

                    if self.use_cuda:
                        pos_u = pos_u.cuda()
                        pos_v = pos_v.cuda()
                        neg_v = neg_v.cuda()

                    self.optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v, self.batch_size)
                    loss.backward()
                    self.optimizer.step()

                    process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(),
                                         self.optimizer.param_groups[0]['lr']))

                #lr = self.initial_lr * (1.0 - 0.1 * j)
                #for param_group in self.optimizer.param_groups:
                #    param_group['lr'] = lr
        except (KeyboardInterrupt, SystemExit):
            first_n = 200
            embeddings = self.get_embedding()
            #self.print_associated_words(embeddings, first_n, sym_dict, mem_offset)
            #self.plot_with_labels(embeddings, first_n, sym_dict, mem_offset)
            return embeddings
        else:
            embeddings = self.get_embedding()
            return embeddings

    def get_embedding(self):
        return self.skip_gram_model.get_embedding(self.use_cuda)


    #printing the word vectors
    def plot_with_labels(self,embeddings, first_n, sym_dict, mem_offset):

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        low_dim_embs = tsne.fit_transform(embeddings[1:first_n, :])
        labels = [ut.get_name(self.data.id2word[i],sym_dict,mem_offset) for i in xrange(1,first_n)]

        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig('tsne.png')


    def print_associated_words(self,embeddings, first_n, sym_dict, mem_offset):
        for w in range(1,first_n):
            embedding = embeddings[w]
            embedding = np.reshape(embedding,(1,-1))
            d = cosine_similarity(embedding, embeddings)[0]
            all_tokens = [self.data.id2word[t] for t in range(self.emb_size)[1:]]
            d = zip(all_tokens,d[1:])
            d = sorted(d, key=lambda x:x[1], reverse=True)
            s = ''
            s += ut.get_name(w,sym_dict,mem_offset) + ' : '
            for k,v in d[:10]:
                s += ut.get_name(k,sym_dict,mem_offset) + ','
            print s

