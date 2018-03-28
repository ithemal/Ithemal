import mysql.connector
import struct
import word2vec as w2v
import argparse
import sys
from mysql.connector import errorcode
import re
import os
from tempfile import gettempdir
import matplotlib
import utilities as ut
import tensorflow as tf
import classification as cl
import numpy as np


matplotlib.use('Agg')


if __name__ == "__main__":

    offsets = ut.read_offsets()
    sym_dict = ut.get_opcode_opnd_dict(opcode_start = offsets[0],opnd_start = offsets[1])
   
    sym_dict[offsets[2]] = 'int_immed'
    sym_dict[offsets[3]] = 'float_immed'

    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str);
    args = parser.parse_args(sys.argv[1:])
    
    cnx = ut.create_connection('training')

    #we have 2 options use the generated tokens as is or build a dataset which will 
    #assign unqiue values to the tokens that are only present in the dataset

    code_data = ut.get_data(cnx,args.format,[])

    token_data = [item for sublist in code_data for item in sublist[0]]
    token_size = 2000 #amount of unique tokens to consider

    cnx.close()

    data, count, dictionary, reverse_dictionary = w2v.build_dataset(token_data, token_size)
  
    final_embeddings = w2v.train_skipgram(data, len(dictionary), reverse_dictionary, sym_dict, offsets[4], 10001)

    embedding_size = final_embeddings.shape[1]
    #embedding_size = 1
    print offsets[0], offsets[4]

    #create the entire dataset from the learnt embeddings
    x = np.ndarray(shape = [len(data),embedding_size]) 
    y = np.ndarray(shape = [len(data),1])

    for i,token in enumerate(token_data):        
        x[i] = final_embeddings[data[i]]
        #x[i] = data[i]
        if token >= offsets[0] and token < offsets[4]:
            y[i] = 1
        else:
            y[i] = 0

    train_x, train_y, test_x, test_y = cl.generate_datasets(x,y,80)
    batch_size = 10000
    cl.train(train_x, train_y, test_x, test_y, batch_size)
    
