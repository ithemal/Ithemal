import sys
sys.path.append('..')

import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
matplotlib.use('Agg')

import common_libs.utilities as ut
import graphs as gr
import numpy as np
import torch

import data.data_additive as dt
import data.data as dt_abs
import models.rnn_models as md
import models.graph_models as mdgr
import models.losses as ls
import models.train as tr


#train using the final value for models (A), (B), (C) - regression
def train_model_regression(data, model, savemodelfile, resultfile, clip=None, lr=0.01):

    train = tr.Train(model, data, epochs = 3, batch_size = 1000, epoch_len_div = 10, lr=lr, clip=clip)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train(savefile=savemodelfile)

    losses = []
    for per_epoch_loss in train.loss:
        for batch_loss in per_epoch_loss:
            losses.append(batch_loss[0])

    results = train.validate(loadfile=savemodelfile, resultfile=resultfile)

    return (losses, results)

if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',type=str,default='text')
    parser.add_argument('--embedfile',action='store',type=str, default='../inputs/code_delim.emb')
    parser.add_argument('--embmode',action='store',type=str, default='none')
    args = parser.parse_args(sys.argv[1:])

    #create the abstract data object
    data = dt_abs.Data()
    data.set_embedding(args.embedfile)
    data.read_meta_data()

    raw_data = torch.load('../saved/ins.data')
    print 'loaded data from file %d' % len(raw_data)
    data.raw_data = raw_data

    data.fields = []

    data.costs = torch.load('../saved/cost20.data')
    cost_prefix = '20'

    embedding_size = data.final_embeddings.shape[1]

    losses = []
    errors = []
    modelnames = []
    eamount = 200

    #task 1 - seq model
    dataToken = dt.DataTokenEmbedding(data)
    dataToken.prepare_data()
    dataToken.generate_datasets()

    print 'running token based RNN ...'
    model_seq = md.ModelSequentialRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=False)
    model_seq.set_learnable_embedding(mode = args.embmode, dictsize = max(dataToken.word2id) + 1, seed = dataToken.final_embeddings)

    model_name = '../saved/paper_seq_additive.mdl'
    result_name = '../results/paper_seq_additive.txt'

    (loss, results) = train_model_regression(dataToken, model_seq, model_name, result_name)
    actual, predicted = results
    errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
    losses.append(loss)
    modelnames.append('Sequential RNN')

    #task 1 - hierarchical model
    dataIns = dt.DataInstructionEmbedding(data)
    dataIns.prepare_data()
    dataIns.generate_datasets()

    print 'running hierarchical model ...'
    model_hierarchical = md.ModelHierarchicalRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=False)
    model_hierarchical.set_learnable_embedding(mode = args.embmode, dictsize = max(dataIns.word2id) + 1, seed = dataIns.final_embeddings)

    model_name = '../saved/paper_hierarchical_additive.mdl'
    result_name = '../results/paper_hierarchical_additive.txt'

    (loss, results) = train_model_regression(dataIns, model_hierarchical, model_name, result_name)
    actual, predicted = results
    errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
    losses.append(loss)
    modelnames.append('Hierarchical RNN')

    print 'running graph neural network model ...'
    dataGraph = dt.DataInstructionEmbedding(data)
    dataGraph.prepare_data()
    dataGraph.generate_datasets()

    model_graph = mdgr.GraphNN(embedding_size=embedding_size,hidden_size=256,num_classes=1)
    model_graph.set_learnable_embedding(mode = args.embmode, dictsize = max(dataGraph.word2id) + 1, seed = dataGraph.final_embeddings)

    model_name = '../saved/paper_graph_additive.mdl'
    result_name = '../results/paper_graph_additive.txt'

    (loss, results) = train_model_regression(dataGraph, model_graph, model_name, result_name)
    actual, predicted = results
    errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
    losses.append(loss)
    modelnames.append('Graph RNN')


    #get only 100 batches

    torch.save(losses, '../results/losses_additive_3.pkl')

    temp_losses = []
    for loss in losses:
        if len(loss) > 100:
            temp_losses.append(loss[:100])
        else:
            temp_losses.append(loss)

    losses = temp_losses



    result_name = '../results/paper_additive.png'
    gr.plot_line_graphs(result_name, losses, modelnames)
    result_name = '../results/paper_additive_error.png'
    gr.plot_line_graphs(result_name, errors, modelnames, xlabel='test case', ylabel='percentage error', title='test set errors', ymin=0, ymax=100)



