import sys
sys.path.append('..')

import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import common.utilities as ut
import numpy as np
import torch
torch.backends.cudnn.enabled = False

import models.graph_models as md
import data.data_cost as dt
import models.losses as ls
import models.train as tr

def graph_cost_classification(database, format, embed_file, savefile, embedding_mode):

    #create database connection
    cnx = ut.create_connection(database)

    data = dt.DataInstructionEmbedding()
    
    data.extract_data(cnx, format, ['time'])
    data.set_embedding(embed_file)
    data.read_meta_data()

    data.prepare_data()
    data.generate_datasets()

    #regression
    num_classes = 1
 
    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = md.GraphNN(embedding_size = embedding_size, hidden_size = 256, num_classes = num_classes)

    model.set_learnable_embedding(mode = embedding_mode, dictsize = max(data.word2id) + 1, seed = data.final_embeddings)

    train = tr.Train(model,data, batch_size = 1000, clip=None, opt='Adam')
           
    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final 
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    train.train(savefile=savefile)
    results = train.validate('../results/realtime_results.txt')
    
    cnx.close()


def graph_model_validation(database, format, embed_file, model_file, embedding_mode):

    #create database connection
    cnx = ut.create_connection(database)

    data = dt.DataInstructionEmbedding()
    
    data.extract_data(cnx, format, ['time','code_text'])
    data.set_embedding(embed_file)
    data.read_meta_data()
    
    data.prepare_data()
    
    data.generate_datasets()

    #regression
    num_classes = 1
 
    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = md.GraphNN(embedding_size = embedding_size, hidden_size = 256, num_classes = num_classes)
    model.set_learnable_embedding(mode = embedding_mode, dictsize = max(data.word2id) + 1, seed = data.final_embeddings)
    train = tr.Train(model,data, batch_size = 1000,  clip=None)
           
    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final 
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    #train.data.test = train.data.test[:10000]

    resultfile = '../results/realtime_results.txt'
    (actual, predicted) = train.validate(resultfile=resultfile, loadfile=model_file)
    
    training_size = len(data.train)


    f = open(resultfile, 'a+')

    for i,result in enumerate(zip(actual,predicted)):
        
        (a,p) = result
        if (abs(a -p) * 100.0 / a) > train.tolerance:
          
            text = data.raw_data[i + training_size][2]
            print a, p
            print text
            f.write('%f, %f\n' % (a,p))
            f.write(text + '\n')
            
    f.close()

    cnx.close()
    



if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--database',action='store',type=str)
    parser.add_argument('--train',action='store',type=bool)

    parser.add_argument('--mode',action='store',type=str,default='learnt')
    parser.add_argument('--embedfile',action='store',type=str,default='../inputs/code_delim.emb')
    parser.add_argument('--savefile',action='store',type=str,default='../saved/graphCost.mdl')
    parser.add_argument('--loadfile',action='store',type=str,default='../saved/graphCost.mdl')

    args = parser.parse_args(sys.argv[1:])

    if args.train:
        graph_cost_classification(args.database, args.format, args.embedfile, args.savefile, args.mode)
        

    graph_model_validation(args.database, args.format, args.embedfile, args.loadfile, args.mode)
    
