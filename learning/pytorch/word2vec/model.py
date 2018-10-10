import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math



class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        #self.linear = nn.Linear(emb_dimension,emb_size)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.normal_(0, 1.0/math.sqrt(self.emb_dimension))

    def forward(self, pos_u, pos_v, neg_v, batch_size):
        #print pos_u, pos_v, neg_v
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        #linear = self.linear(emb_u)
        #return nn.functional.cross_entropy(linear,pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return  -1 * (torch.sum(score)+torch.sum(neg_score)) / batch_size

    def get_embedding(self, use_cuda):
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data
        else:
            embedding = self.u_embeddings.weight.data

        #normalize the embeddings
        norm = torch.sqrt(torch.sum(torch.mul(embedding,embedding),dim = 1,keepdim=True))
        #print norm.shape, embedding.shape

        embedding = embedding / norm


        return embedding.numpy()




