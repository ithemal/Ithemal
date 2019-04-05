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

import data.data_span as dt
import data.data as dt_abs
import models.rnn_models as md
import models.graph_models as md_gr
import models.losses as ls
import models.train as tr


#train using the final value for models (A), (B), (C) - regression
def train_model_regression(data, model, savemodelfile, resultfile, clip=10):

    train = tr.Train(model, data, epochs = 3, batch_size = 1000, epoch_len_div = 2, lr = 0.001, clip=clip)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train(savefile=savemodelfile)

    losses = []
    for per_epoch_loss in train.loss:
        for batch_loss in per_epoch_loss:
            losses.append(batch_loss[0])

    results = train.validate(resultfile=resultfile)

    return (losses, results)


def train_model_classification(data, model, savemodelfile, resultfile, clip=10):

    train = tr.Train(model, data, epochs = 3, batch_size = 1000, epoch_len_div = 2, lr = 0.001, clip=clip)

    train.loss_fn = ls.cross_entropy_loss
    train.print_fn = train.print_max
    train.correct_fn = train.correct_classification
    train.num_losses = 1

    train.train(savefile=savemodelfile)

    losses = []
    for per_epoch_loss in train.loss:
        for batch_loss in per_epoch_loss:
            losses.append(batch_loss[0])

    results = train.validate(resultfile=resultfile)

    return (losses, results)

if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str,default='../inputs/code_delim.emb')
    parser.add_argument('--mode',action='store',type=str,default='learnt')
    args = parser.parse_args(sys.argv[1:])

    #create the abstract data object
    data = dt_abs.Data()
    data.set_embedding(args.embed_file)
    data.read_meta_data()

    raw_data = torch.load('../saved/ins.data')
    print 'loaded data from file %d' % len(raw_data)

    #remove all basic blocks < 5 instructions
    final_raw_data = []
    for row in raw_data:
        block = ut.create_basicblock(row[0])
        if block.num_instrs() >= 5:
            final_raw_data.append(row)

    print 'after removing small bbs - size - ' + str(len(final_raw_data))
    data.raw_data = final_raw_data
    data.fields = []


    embedding_size = data.final_embeddings.shape[1]

    losses_regress = []
    losses_classification = []
    errors_regress = []
    errors_classification = []
    eamount = 100
    modelnames = ['graph >= 5 ins']

    print 'runnning for large bbs...'
    #task - regression
    print 'regression model D...'
    dataIns = dt.DataInstructionEmbedding(data)
    #dataIns.costs = torch.load('../saved/cost20.data')
    dataIns.prepare_data()
    dataIns.generate_datasets()


    model = md_gr.GraphNN(embedding_size=embedding_size,hidden_size=256,num_classes=1)

    model.set_learnable_embedding(mode = args.mode, dictsize = max(dataIns.word2id) + 1, seed = dataIns.final_embeddings)


    model_name = '../saved/modelDspanregressbigbb.mdl'
    result_name = '../results/modelDspanregressbigbb.txt'

    (loss, results) = train_model_regression(dataIns, model, model_name, result_name)
    actual, predicted = results
    errors_regress.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
    losses_regress.append(loss)

    result_name = '../results/spanregressbigbb.png'
    gr.plot_line_graphs(result_name, losses_regress, modelnames)
    result_name = '../results/spanregressbigbbtesterror.png'
    gr.plot_line_graphs(result_name, errors_regress, modelnames, xlabel='test case', ylabel='percentage error', title='test set errors', ymin=0, ymax=100)


    #task - classification
    print 'classification model D...'
    num_classes = dataIns.prepare_for_classification()
    dataIns.generate_datasets()

    model = md_gr.GraphNN(embedding_size=embedding_size,hidden_size=256,num_classes=num_classes)

    model.set_learnable_embedding(mode = args.mode, dictsize = max(dataIns.word2id) + 1, seed = dataIns.final_embeddings)

    model_name = '../saved/modelDspanclassificationbigbb.mdl'
    result_name = '../results/modelDspanclassificationbigbb.txt'

    (loss, results) = train_model_classification(dataIns, model, model_name, result_name)
    actual, predicted = results
    errors_classification.append(ut.get_percentage_error(predicted[:eamount], actual[:eamount]))
    losses_classification.append(loss)

    result_name = '../results/spanclassificationbigbb.png'
    gr.plot_line_graphs(result_name, losses_classification, modelnames)
    result_name = '../results/spanclassificationbigbbtesterror.png'
    gr.plot_line_graphs(result_name, errors_classification, modelnames, xlabel='test case', ylabel='percentage error', title='test set errors', ymin=0, ymax=100)


