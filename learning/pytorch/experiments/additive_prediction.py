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
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str, default='../inputs/code_delim.emb')
    parser.add_argument('--cost_dist',action='store',type=int)
    parser.add_argument('--run_model',action='store',type=int, default=0)
    parser.add_argument('--mode',action='store',type=str,default='learnt')
    args = parser.parse_args(sys.argv[1:])

    #create the abstract data object
    data = dt_abs.Data()
    data.set_embedding(args.embed_file)
    data.read_meta_data()

    raw_data = torch.load('../saved/ins.data')
    print 'loaded data from file %d' % len(raw_data)
    data.raw_data = raw_data
    data.fields = []

    cost_prefix = ''
    if args.cost_dist == 20:
        data.costs = torch.load('../saved/cost20.data')
        cost_prefix = str(args.cost_dist)
    elif args.cost_dist == 50:
        data.costs = torch.load('../saved/cost50.data')
        cost_prefix = str(args.cost_dist)
    elif args.cost_dist == 100:
        data.costs = torch.load('../saved/cost100.data')
        cost_prefix = str(args.cost_dist)

    embedding_size = data.final_embeddings.shape[1]

    losses = []
    errors = []
    modelnames = []
    eamount = 100

    #task 1 - model A
    dataToken = dt.DataTokenEmbedding(data)
    dataToken.prepare_data()
    dataToken.generate_datasets()

    if args.run_model == 0 or args.run_model == 1:
        print 'running model A...'
        modelA = md.ModelSequentialRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=False)

        modelA.set_learnable_embedding(mode = args.mode, dictsize = max(dataToken.word2id) + 1, seed = dataToken.final_embeddings)

        model_name = '../saved/modelAadd' + cost_prefix + '.mdl'
        result_name = '../results/modelAadd' + cost_prefix + '.txt'

        (loss, results) = train_model_regression(dataToken, modelA, model_name, result_name)
        actual, predicted = results
        errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
        losses.append(loss)
        modelnames.append('sequential')

    #task 2 - model B
    for item in dataToken.data:
        item.y = item.cost
    dataToken.generate_datasets()

    if args.run_model == 0 or args.run_model == 2:
        print 'running model B....'
        modelB = md.ModelSequentialRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=True)

        modelB.set_learnable_embedding(mode = args.mode, dictsize = max(dataToken.word2id) + 1, seed = dataToken.final_embeddings)

        model_name = '../saved/modelBadd' + cost_prefix + '.mdl'
        result_name = '../results/modelBadd' + cost_prefix + '.txt'

        (loss, results) = train_model_regression(dataToken, modelB, model_name, result_name)
        actual, predicted = results
        errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
        losses.append(loss)
        modelnames.append('sequential-intermediate')


    #task 3 - model C
    dataIns = dt.DataInstructionEmbedding(data)
    dataIns.prepare_data()
    dataIns.generate_datasets()

    if args.run_model == 0 or args.run_model == 3:
        print 'running model C...'
        modelC = md.ModelHierarchicalRNN(embedding_size=embedding_size,hidden_size=256,num_classes=1,intermediate=False)

        modelC.set_learnable_embedding(mode = args.mode, dictsize = max(dataIns.word2id) + 1, seed = dataIns.final_embeddings)


        model_name = '../saved/modelCadd' + cost_prefix + '.mdl'
        result_name = '../results/modelCadd' + cost_prefix + '.txt'

        (loss, results) = train_model_regression(dataIns, modelC, model_name, result_name)
        actual, predicted = results
        errors.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
        losses.append(loss)
        modelnames.append('hierarchical')

    result_name = '../results/add' + cost_prefix + '.png'
    gr.plot_line_graphs(result_name, losses, modelnames)
    result_name = '../results/addtesterror' + cost_prefix + '.png'
    gr.plot_line_graphs(result_name, errors, modelnames, xlabel='test case', ylabel='percentage error', title='test set errors', ymin=0, ymax=100)



