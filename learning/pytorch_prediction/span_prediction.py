import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import sys
import common.utilities as ut
import numpy as np
import torch
import models.rnn_models as md_rnn
import models.graph_models as md_gr
import models.losses as ls
import models.train as tr
import data.data_span as dt

def graph_span_classification(database, format, embed_file):

    #create database connection
    cnx = ut.create_connection(database)

    data = dt.DataInstructionEmbedding()

    data.prepare_data(cnx, format, embed_file)

    data.generate_datasets()

    data.count_span_dist(data.train)
    data.count_span_dist(data.test)

    #prepare for classification
    num_classes = data.prepare_for_classification()
    print 'classes ' + str(num_classes)
    data.generate_datasets()

    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    
    #train
    model = md_gr.GraphNN(embedding_size, num_classes)
    train = tr.Train(model,data, batch_size = 1000)
    
    train.loss_fn = ls.cross_entropy_loss
    train.print_fn = train.print_max
    train.correct_fn = train.correct_classification
    train.num_losses = 1
    
    train.train()
    train.validate('span.txt')
    
    cnx.close()


def inst_span_classification(database, format, embed_file):

    #create database connection
    cnx = ut.create_connection(database)

    data = dt.DataInstructionEmbedding()

    data.prepare_data(cnx, format, embed_file)

    data.generate_datasets()

    data.count_span_dist(data.train)
    data.count_span_dist(data.test)

    #prepare for classification
    num_classes = data.prepare_for_classification()
    print 'classes ' + str(num_classes)
    data.generate_datasets()

    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    
    #train
    model = md_rnn.ModelInstructionEmbeddingClassification(embedding_size, num_classes)
    train = tr.Train(model,data, batch_size = 1000)
    
    train.loss_fn = ls.cross_entropy_loss
    train.print_fn = train.print_max
    train.correct_fn = train.correct_classification
    train.num_losses = 1
    
    train.train()
    train.validate('span.txt')
    
    cnx.close()



if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    
    graph_span_classification('static0512', args.format, args.embed_file)
