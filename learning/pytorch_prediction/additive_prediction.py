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

import models.rnn_models as Model
import data.data_numins as Data

if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    #create database connection
    cnx = ut.create_connection('static0512')

    data = Data.DataInstructionEmbedding()

    data.prepare_data(cnx, args.format, args.embed_file)


    # we need learning curves for various settings + evaluation of them
    # print len(data.x)
    # #we can filter certain data out here
    # if isinstance(data, rnn.DataInstructionEmbedding):
    #     print 'filtering....'
    #     for i, x in enumerate(data.x):
    #         if len(x) < 5:
    #             del data.x[i]
    #             del data.y[i]
    #             del data.cost[i]
     
    # print len(data.x)

    data.generate_datasets()

    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = Model.ModelInstructionEmbedding(embedding_size)
    train = Model.Train(model,data)

    train.train(train.mse_loss,1)
    train.validate(train.mse_loss,1)

    #train.train()
    #train.validate()


    # #ok now we need to check intermediate values
    # data.test_x = data.x
    # data.test_y = data.cost

    # embedding_size = data.final_embeddings.shape[1]
    # testmodel = RNN.ModelInstructionAggregate(embedding_size)
    # # testmodel.copy(model)

    # testtrain = rnn.Train(testmodel,data)
    # testtrain.train_with_rankloss()
    # testtrain.validate_with_rankloss()
    
    cnx.close()
    
