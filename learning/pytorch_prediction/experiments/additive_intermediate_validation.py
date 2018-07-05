import sys
sys.path.append('..')

import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
matplotlib.use('Agg')

import common.utilities as ut
import common.graphs as gr
import numpy as np
import torch


import data.data_additive as dt
import data.data as dt_abs
import models.rnn_models as md
import models.losses as ls
import models.train as tr


#check if intermediate results are predicted properly
def validate_regression(data, model, loadfile, resultfile):

    train = tr.Train(model, data)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.validate(loadfile=loadfile, resultfile=resultfile)    


if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    #create the abstract data object
    data = dt_abs.Data()
    data.set_embedding(args.embed_file)
    data.read_meta_data()

    raw_data = torch.load('../saved/ins.data')
    print 'loaded data from file %d' % len(raw_data)
    data.raw_data = raw_data
    data.fields = []

    embedding_size = data.final_embeddings.shape[1]


    #task 1 - model A on intermediate results
    dataToken = dt.DataTokenEmbedding(data)
    dataToken.prepare_data()
    for item in dataToken.data:
        item.y = item.cost
    dataToken.generate_datasets()

    modelA = md.ModelSequentialRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=True)
    
    validate_regression(dataToken, modelA, '../saved/modeAins.mdl', '../results/modelAinsinter.txt')

   
    #task 2 - model C on intermediate results
    dataIns = dt.DataInstructionEmbedding(data)
    dataIns.prepare_data()
    for item in dataToken.data:
        item.y = item.cost
    dataIns.generate_datasets()

    modelC = md.ModelHierarchicalRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=True)
    
    validate_regression(dataIns, modelC, '../saved/modeCins.mdl', '../results/modelCinsinter.txt')



