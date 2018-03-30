import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import sys
import utilities as ut
import tensorflow as tf
import binary_classification as bc
import numpy as np

matplotlib.use('Agg')


if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    args = parser.parse_args(sys.argv[1:])

    #create database connection
    cnx = ut.create_connection('training')

    data = bc.Data()
    model = bc.Model(data)
    
    data.extract_and_prepare_data(cnx,args.format)
    data.generate_datasets()
    
    model.generate_model()
    params = model.train_model()
    model.test_model(params)

    cnx.close()
    
