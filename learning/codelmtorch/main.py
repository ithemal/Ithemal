import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import sys
import common.utilities as ut
import numpy as np

matplotlib.use('Agg')

import rnn_numins as rnn


if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--embed_file',action='store',type=str)
    args = parser.parse_args(sys.argv[1:])

    #create database connection
    cnx = ut.create_connection('costmodel')

    data = rnn.DataInstructionEmbedding()
    data.extract_data(cnx,args.format,args.embed_file)
    data.prepare_data()
    data.generate_datasets()

    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = rnn.ModelInstructionEmbedding(embedding_size)
    train = rnn.Train(model,data)

    train.train()
    train.validate()

    cnx.close()
    
