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
import time
import torch
import torch.multiprocessing as mp
torch.backends.cudnn.enabled = False

import models.graph_models as md
import data.data_cost as dt
import models.losses as ls
import models.train as tr
from tqdm import tqdm
from mpconfig import MPConfig
from typing import List

def save_data(database, config, format, savefile, arch):

    cnx = ut.create_connection_from_config(database=database, config_file=config)

    data = dt.DataInstructionEmbedding()

    data.extract_data(cnx, format, ['code_id','code_intel'])
    data.get_timing_data(cnx, arch)

    torch.save(data.raw_data, savefile)

def get_partition_splits(n_datapoints, n_trainers, split_distr):
    assert abs(sum(split_distr) - 1) < 1e-4
    assert all(elem >= 0 for elem in split_distr)

    idx = 0
    for frac in split_distr:
        split_size = int((n_datapoints / n_trainers) * frac)
        for tr in range(n_trainers):
            yield (idx, idx + split_size)
            idx += split_size
    yield (idx, n_datapoints)


def graph_model_learning(data_savefile, embed_file, savefile, embedding_mode, split_dist, no_decay_procs, initial_lr):
    # type: (str, str, str, str, List[float], bool, float) -> None
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

    lr = initial_lr
    train = tr.Train(model, data, batch_size=args.batch_size, clip=None, opt='Adam', lr=lr)

    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    restored_epoch = -1
    restored_batch_num = -1

    if args.loadfile is not None:
        state_dict = train.load_checkpoint(args.loadfile)
        restored_epoch = state_dict['epoch']
        restored_batch_num = state_dict['batch_num']
        print('starting from a checkpointed state... epoch {} batch_num {}'.format(
            restored_epoch,
            restored_batch_num,
        ))

    model.share_memory()

    start_time = time.time()

    def run_training(q, epoch_idx, rank, loss_report_func):
        while True:
            part = q.get()
            if part is None:
                return
            train(epoch_idx, rank, part, savefile, start_time, loss_report_func)

    def loss_report_func(ep_no, loss_q):
        # type: (mp.Queue) -> None
        format_loss = lambda l: 'Epoch {}, Loss {:.2}'.format(ep_no, l)
        pbar = tqdm(desc=format_loss('??'), total=len(data.train))
        ema_loss = None
        while True:
            item = loss_q.get()
            if item is None:
                return
            loss = item.loss
            if ema_loss is None:
                ema_loss = loss
            else:
                ema_loss = ema_loss * 0.99 + loss * 0.01

            pbar.update(item.n_items)
            pbar.set_description(format_loss(ema_loss))

    n_trainers = args.trainers

    for i in range(args.epochs):
        mp_config = MPConfig(n_trainers, args.threads)
        partitions = list(get_partition_splits(len(data.train), mp_config.trainers, split_dist))
        partitions += [None] * mp_config.trainers

        processes = []
        i = restored_epoch + i + 1

        partition_queue = mp.Queue()
        loss_q = mp.Queue()

        mp.Process(target=loss_report_func, args=(i, loss_q)).start()

        with mp_config:
            for rank in range(mp_config.trainers):
                mp_config.set_env(rank)

                m_args = (partition_queue, i, rank, loss_q.put)
                p = mp.Process(target=run_training, args=m_args)
                p.start()
                print("Starting process %d" % (rank,))
                processes.append(p)

        for split in partitions:
            partition_queue.put(split)

        for p in processes:
            p.join()

        loss_q.put(None)

        if args.savefile is not None:
            train.save_checkpoint(i, 0, args.savefile)

        lr /= 10
        train.set_lr(lr)
        if not no_decay_procs:
            n_trainers -= 1

    resultfile = os.path.join(
        os.environ['ITHEMAL_HOME'],
        'learning',
        'pytorch',
        'results',
        'realtime_results.txt',
    )
    results = train.validate(resultfile)

def graph_model_benchmark(data_savefile, embed_file, embedding_mode, n_examples):
    # type: (str, str, str, int) -> None

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

    train = tr.Train(model, data, batch_size=args.batch_size, clip=None, opt='Adam', lr=0.01)

    #defining losses, correctness and printing functions
    train.loss_fn = ls.mse_loss
    train.print_fn = train.print_final
    train.correct_fn = train.correct_regression
    train.num_losses = 1

    model.share_memory()

    mp_config = MPConfig(args.trainers, args.threads)
    partition_size = n_examples // args.trainers

    processes = []

    start_time = time.time()

    with mp_config:
        for rank in range(mp_config.trainers):
            mp_config.set_env(rank)

            partition = (rank * partition_size, (rank + 1) * partition_size)

            p = mp.Process(target=train, args=(0, rank, partition))
            p.start()
            print("Starting process %d" % (rank,))
            processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    print('Time to process {} examples: {:.2} seconds'.format(
        n_examples,
        end_time - start_time,
    ))

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


    train = tr.Train(model, data,  batch_size = args.batch_size, clip=None, opt='Adam', lr=0.01)

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
    parser.add_argument('--n-examples', type=int, default=1000)
    parser.add_argument('--no-decay-procs', action='store_true', default=False)
    parser.add_argument('--split-dist', nargs='+', type=float,
                        default=[0.5, 0.25, 0.125, .0625, .0625])
    parser.add_argument('--initial-lr', type=float, default=0.01)


    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'save':
        save_data(args.database, args.config, args.format, args.savedatafile, args.arch)
    elif args.mode == 'train':
        graph_model_learning(args.savedatafile, args.embedfile, args.savefile, args.embmode, args.split_dist, args.no_decay_procs, args.initial_lr)
    elif args.mode == 'validate':
        graph_model_validation(args.savedatafile, args.embedfile, args.loadfile, args.embmode)
    elif args.mode == 'predict':
        graph_model_gettiming(args.database, args.config, args.format, args.savedatafile, args.embedfile, args.loadfile, args.embmode, args.arch)
    elif args.mode == 'benchmark':
        graph_model_benchmark(args.savedatafile, args.embedfile, args.embmode, args.n_examples)
