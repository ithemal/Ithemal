import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import sys
import common.utilities as ut
import numpy as np
import torch

matplotlib.use('Agg')

import data.data_numins as dt
import models.rnn_models as md
import models.losses as ls
import models.train as tr


def test_instruction_embedding(database, format, embed_file):

    cnx = ut.create_connection(database)

    #data for token based models
    data = dt.DataInstructionEmbedding()
    data.prepare_data(cnx, format, embed_file)
    data.generate_datasets()

    #model
    embedding_size = data.final_embeddings.shape[1]
    model = md.ModelInstructionEmbedding(embedding_size)
    train = tr.Train(model, data, epochs = 3, batch_size = 1000)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train()
    train.validate('instruction_cost.txt')

    aggregate_model = md.ModelInstructionAggregate(embedding_size)
    aggregate_model.copy(model)

    #change the dataset
    for item in data.train:
        item.y = item.cost
    for item in data.test:
        item.y = item.cost
        
    train.model = aggregate_model

    train.validate('instruction_aggregate.txt')
    cnx.close()


def test_token_aggregate(database, format, embed_file):

    cnx = ut.create_connection(database)

    #data for token based models
    data = dt.DataAggregateCost()
    data.prepare_data(cnx, format, embed_file)
    data.generate_datasets()

    #model
    embedding_size = data.final_embeddings.shape[1]
    model = md.ModelAggregate(embedding_size)
    train = tr.Train(model, data, batch_size = 100)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train()
    train.validate('token_aggregate_cost.txt')

    cnx.close()
    

def test_token_final(database, format, embed_file):

    """
    token embedding + final timing value
    """
    cnx = ut.create_connection(database)

    #data for token based models
    data = dt.DataFinalCost()
    data.prepare_data(cnx, format, embed_file)
    data.generate_datasets()

    #model
    embedding_size = data.final_embeddings.shape[1]
    model = md.ModelFinalHidden(embedding_size)
    train = tr.Train(model, data, batch_size = 100)

    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train()
    train.validate('token_final_cost.txt')

    cnx.close()
    

if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    test_instruction_embedding('static0512', args.format, args.embed_file)


