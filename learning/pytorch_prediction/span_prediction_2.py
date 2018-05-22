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
import data.data_span as Data


if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    #create database connection
    cnx = ut.create_connection('static0512')

    data = Data.DataInstructionEmbedding()

    #data.update_span(cnx, args.format)
    #exit()

    data.prepare_data(cnx,args.format,args.embed_file)

    data.generate_datasets()

    data.count_span_dist(data.train_y)
    data.count_span_dist(data.test_y)

    
    num_classes = data.prepare_for_classification()

    data.generate_datasets()
    

    
    #
    # #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = Model.ModelInstructionEmbeddingClassification(embedding_size, num_classes)
    train = Model.Train(model,data, batch_size = 100)

    train.train(train.cross_entropy_loss,1)
    train.validate(train.cross_entropy_loss,1, train.print_max, 'span_basic.txt')
    
    cnx.close()
    
