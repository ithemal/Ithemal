import sys
import os
sys.path.append(os.environ['ITHEMAL_HOME'] + '/learning/pytorch')
import mysql.connector
import struct
import word2vec as w2v
import argparse
import matplotlib
import common_libs.utilities as ut
import numpy as np
import torch
import torch.multiprocessing as mp
torch.backends.cudnn.enabled = False

import models.graph_models as md
import data.data_cost as dt
import models.losses as ls
import models.train as tr
from tqdm import tqdm
from mpconfig import MPConfig

def save_data(database, config, format, savefile, arch):

    cnx = ut.create_connection_from_config(database=database, config_file=config)

    data = dt.DataInstructionEmbedding()

    data.extract_data(cnx, format, ['code_id','code_intel'])
    data.get_timing_data(cnx, arch)

    torch.save(data.raw_data, savefile)
    
def graph_model_learning(data_savefile, embed_file, savefile, embedding_mode):


    data = dt.DataInstructionEmbedding()

    data.raw_data = torch.load(data_savefile)
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

    train = tr.Train(model, data, batch_size = args.batch_size, clip=None, opt='Adam', lr = 0.01)
           
    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    restored_epoch = -1
    restored_batch_num = -1

    if args.loadfile != None:
        (restored_epoch, restored_batch_num) = train.load_checkpoint(args.loadfile)
        print 'starting from a checkpointed state... epoch %d batch_num %d' % (restored_epoch, restored_batch_num)

    model.share_memory()
        
    mp_config = MPConfig(args.trainers, args.threads)

    partition_size = len(data.train) // mp_config.trainers
    
    processes = []
  
    for i in range(args.epochs):
        i = restored_epoch + 1

        with mp_config :
            for rank in range(mp_config.trainers):
                
                mp_config.set_env(rank)

                # XXX: this drops data on the floor, namely the tail of the file
                # if the examples are not evenly divisble by the number of trainers
                partition = (rank * partition_size, (rank + 1) * partition_size)

                p = mp.Process(target=train, args=(i, rank, partition))
                p.start()
                print("Starting process %d" % (rank,))
                processes.append(p)
         
        for p in processes:
            p.join()
        
        if args.savefile is not None :
            train.save_checkpoint(i, 0, args.savefile)

    resultfile = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/results/realtime_results.txt'
    results = train.validate(resultfile)

def graph_model_validation(data_savefile, embed_file, model_file, embedding_mode):

    data = dt.DataInstructionEmbedding()
    data.raw_data = torch.load(data_savefile)
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

    resultfile = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/results/realtime_results.txt'
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

def graph_model_gettiming(database, config, format, data_savefile, embed_file, model_file, embedding_mode, arch):

    cnx = ut.create_connection(database=database, config_file=config)

    data = dt.DataInstructionEmbedding()
    data.raw_data = torch.load(data_savefile)
    data.set_embedding(embed_file)
    data.read_meta_data()

    data.prepare_data()
    data.test = data.data #all data are test data now

    #regression
    num_classes = 1

    #get the embedding size
    embedding_size = data.final_embeddings.shape[1]
    model = md.GraphNN(embedding_size = embedding_size, hidden_size = 256, num_classes = num_classes)
    model.set_learnable_embedding(mode = embedding_mode, dictsize = max(data.word2id) + 1, seed = data.final_embeddings)


    train = tr.Train(model, data,  batch_size = args.batch_size, clip=None, opt='Adam', lr = 0.01)    

    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    resultfile = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/results/realtime_results.txt'
    (actual, predicted) = train.validate(resultfile=resultfile, loadfile=model_file)


    #ok now enter the results in the database
    for i, data in enumerate(tqdm(data.test)):

        code_id = data.code_id
        kind = 'predicted'
        time = predicted[i]


        sql = 'INSERT INTO times (code_id, arch, kind, time) VALUES ('
        sql += str(code_id) + ','
        sql += str(arch) + ','
        sql += '\'' + kind + '\','
        sql += str(int(round(time))) + ')'

        ut.execute_query(cnx, sql, False)
        cnx.commit()


    cnx.close()


if __name__ == "__main__":

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',action='store',default='text',type=str)
    parser.add_argument('--mode',action='store',type=str)

    parser.add_argument('--savedatafile',action='store',type=str,default='../inputs/data/timing.data')
    parser.add_argument('--embmode',action='store',type=str,default='learnt')
    parser.add_argument('--embedfile',action='store',type=str,default='../inputs/embeddings/code_delim.emb')
    parser.add_argument('--savefile',action='store',type=str,default='../inputs/models/graphCost.mdl')
    parser.add_argument('--loadfile',action='store',type=str,default=None)
    parser.add_argument('--arch',action='store',type=int, default=1)

    parser.add_argument('--database',action='store',type=str)
    parser.add_argument('--config',action='store',type=str)

    parser.add_argument('--epochs',action='store',type=int,default=1)
    
    parser.add_argument('--trainers',action='store',type=int,default=1)
    parser.add_argument('--threads',action='store',type=int, default=4)
    parser.add_argument('--batch-size',action='store',type=int, default=100)


    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'save':
        save_data(args.database, args.config, args.format, args.savedatafile, args.arch)
    elif args.mode == 'train':
        graph_model_learning(args.savedatafile, args.embedfile, args.savefile, args.embmode)
    elif args.mode == 'validate':
        graph_model_validation(args.savedatafile, args.embedfile, args.loadfile, args.embmode)
    elif args.mode == 'predict':
        graph_model_gettiming(args.database, args.config, args.format, args.savedatafile, args.embedfile, args.loadfile, args.embmode, args.arch)



