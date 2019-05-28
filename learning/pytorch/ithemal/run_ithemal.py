import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import argparse
import time
import torch
import torch.multiprocessing as mp
torch.backends.cudnn.enabled = False
from utils import messages
import models.losses as ls
import models.train as tr
from tqdm import tqdm
from mpconfig import MPConfig
from typing import Callable, List, Optional, Iterator, Tuple, NamedTuple, Union
import random
import Queue
from ithemal_utils import *
import training
import pandas as pd
import common_libs.utilities as ut

def graph_model_benchmark(base_params, benchmark_params):
    # type: (BaseParameters, BenchmarkParameters) -> None
    data = load_data(base_params)
    model = load_model(base_params, data)

    train = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=benchmark_params.batch_size, clip=None, opt=tr.OptimizerType.ADAM_PRIVATE, lr=0.01,
    )

    model.share_memory()

    mp_config = MPConfig(benchmark_params.threads)
    partition_size = benchmark_params.examples // benchmark_params.trainers

    processes = []

    start_time = time.time()

    with mp_config:
        for rank in range(benchmark_params.trainers):
            mp_config.set_env(rank)

            partition = (rank * partition_size, (rank + 1) * partition_size)

            p = mp.Process(target=train, args=(rank, partition))
            p.daemon = True
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    print('Time to process {} examples: {} seconds'.format(
        benchmark_params.examples,
        end_time - start_time,
    ))

def graph_model_validate(base_params, model_file, iaca_only):
    # type: (BaseParameters, str, bool) -> None
    data = load_data(base_params)
    if iaca_only:
        cnx = ut.create_connection()
        legal_code_ids = set(
            pd.read_sql('SELECT time_id, code_id FROM times WHERE kind="iaca"', cnx)
            .set_index('time_id')
            .code_id
        )
        data.test = [datum for datum in data.test if datum.code_id in legal_code_ids]
    model = load_model(base_params, data)

    train = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=1000, clip=None, predict_log=base_params.predict_log,
    )

    resultfile = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/results/realtime_results.txt'
    (actual, predicted) = train.validate(resultfile=resultfile, loadfile=model_file)

def graph_model_dump(base_params, model_file):
    # type: (BaseParameters, str) -> None
    data = load_data(base_params)
    model = load_model(base_params, data)
    dump_model_and_data(model, data, model_file)

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--embed-mode', help='The embedding mode to use (default: none)', default='none')
    parser.add_argument('--embed-file', help='The embedding file to use (default: code_delim.emb)',
                        default=os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch', 'inputs', 'embeddings', 'code_delim.emb'))
    parser.add_argument('--embed-size', help='The size of embedding to use (default: 256)', default=256, type=int)
    parser.add_argument('--hidden-size', help='The size of hidden layer to use (default: 256)', default=256, type=int)
    parser.add_argument('--no-mem', help='Remove all instructions with memory', default=False, action='store_true')

    # edge/misc arguments
    parser.add_argument('--random-edge-freq', type=float, default=0.0, help='The fraction of instructions to add an additional random forward edge to (can be >1)')
    parser.add_argument('--no-residual', default=False, action='store_true', help='Don\'t use a residual model in Ithemal')
    parser.add_argument('--no-dag-rnn', default=False, action='store_true', help='Don\'t use the DAG-RNN model in Ithemal')
    parser.add_argument('--predict-log', action='store_true', default=False, help='Predict the log of the time')
    parser.add_argument('--linear-embeddings', action='store_true', default=False, help='Use linear embeddings instead of LSTM')

    parser.add_argument('--use-rnn', action='store_true', default=False)
    rnn_type_group = parser.add_mutually_exclusive_group()
    rnn_type_group.add_argument('--rnn-normal', action='store_const', const=md.RnnType.RNN, dest='rnn_type')
    rnn_type_group.add_argument('--rnn-lstm', action='store_const', const=md.RnnType.LSTM, dest='rnn_type')
    rnn_type_group.add_argument('--rnn-gru', action='store_const', const=md.RnnType.GRU, dest='rnn_type')
    parser.set_defaults(rnn_type=md.RnnType.LSTM)

    rnn_hierarchy_type_group = parser.add_mutually_exclusive_group()
    rnn_hierarchy_type_group.add_argument('--rnn-token', action='store_const', const=md.RnnHierarchyType.NONE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-dense', action='store_const', const=md.RnnHierarchyType.DENSE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-multiscale', action='store_const', const=md.RnnHierarchyType.MULTISCALE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-linear-model', action='store_const', const=md.RnnHierarchyType.LINEAR_MODEL, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-mop', action='store_const', const=md.RnnHierarchyType.MOP_MODEL, dest='rnn_hierarchy_type')
    parser.set_defaults(rnn_hierarchy_type=md.RnnHierarchyType.MULTISCALE)

    parser.add_argument('--rnn-skip-connections', action='store_true', default=False)
    parser.add_argument('--rnn-learn-init', action='store_true', default=False)
    parser.add_argument('--rnn-connect-tokens', action='store_true', default=False)

    dag_nonlinearity_group = parser.add_mutually_exclusive_group()
    dag_nonlinearity_group.add_argument('--dag-relu-nonlinearity', action='store_const', const=md.NonlinearityType.RELU, dest='dag_nonlinearity')
    dag_nonlinearity_group.add_argument('--dag-tanh-nonlinearity', action='store_const', const=md.NonlinearityType.TANH, dest='dag_nonlinearity')
    dag_nonlinearity_group.add_argument('--dag-sigmoid-nonlinearity', action='store_const', const=md.NonlinearityType.SIGMOID, dest='dag_nonlinearity')
    parser.set_defaults(dag_nonlinearity=None)
    parser.add_argument('--dag-nonlinearity-width', help='The width of the final nonlinearity (default: 128)', default=128, type=int)
    parser.add_argument('--dag-nonlinear-before-max', action='store_true', default=False)

    data_dependency_group = parser.add_mutually_exclusive_group()
    data_dependency_group.add_argument('--linear-dependencies', action='store_true', default=False)
    data_dependency_group.add_argument('--flat-dependencies', action='store_true', default=False)

    dag_reduction_group = parser.add_mutually_exclusive_group()
    dag_reduction_group.add_argument('--dag-add-reduction', action='store_const', const=md.ReductionType.ADD, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-max-reduction', action='store_const', const=md.ReductionType.MAX, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-mean-reduction', action='store_const', const=md.ReductionType.MEAN, dest='dag_reduction')
    dag_reduction_group.add_argument('--dag-attention-reduction', action='store_const', const=md.ReductionType.ATTENTION, dest='dag_reduction')
    parser.set_defaults(dag_reduction=md.ReductionType.MAX)

    def add_edge_ablation(ablation):
        # type: (EdgeAblationType) -> None
        parser.add_argument('--{}'.format(ablation.value), action='append_const', dest='edge_ablations', const=ablation)

    add_edge_ablation(EdgeAblationType.TRANSITIVE_REDUCTION)
    add_edge_ablation(EdgeAblationType.TRANSITIVE_CLOSURE)
    add_edge_ablation(EdgeAblationType.ADD_LINEAR_EDGES)
    add_edge_ablation(EdgeAblationType.ONLY_LINEAR_EDGES)
    add_edge_ablation(EdgeAblationType.NO_EDGES)

    sp = parser.add_subparsers(dest='subparser')

    train = sp.add_parser('train', help='Train an ithemal model')
    train.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    train.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    train.add_argument('--load-file', help='Start by loading the provided model')

    train.add_argument('--batch-size', type=int, default=4, help='The batch size to use in train')
    train.add_argument('--epochs', type=int, default=3, help='Number of epochs to run for')
    train.add_argument('--trainers', type=int, default=4, help='Number of trainer processes to use')
    train.add_argument('--threads', type=int,  default=4, help='Total number of PyTorch threads to create per trainer')
    train.add_argument('--decay-trainers', action='store_true', default=False, help='Decay the number of trainers at the end of each epoch')
    train.add_argument('--weight-decay', type=float, default=0, help='Coefficient of weight decay (L2 regularization) on model')
    train.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate')
    train.add_argument('--decay-lr', action='store_true', default=False, help='Decay the learning rate at the end of each epoch')
    train.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter for SGD')
    train.add_argument('--nesterov', action='store_true', default=False, help='Use Nesterov momentum')
    train.add_argument('--weird-lr', action='store_true', default=False, help='Use unusual LR schedule')
    train.add_argument('--lr-decay-rate', default=1.2, help='LR division rate', type=float)

    split_group = train.add_mutually_exclusive_group()
    split_group.add_argument(
        '--split-dist', action='store_const', const=[0.5, 0.25, 0.125, .0625, .0625],
        help='Split data partitions between trainers via a distribution',
    )
    split_group.add_argument('--split-size', type=int, help='Partitions of a fixed size')

    optimizer_group = train.add_mutually_exclusive_group()
    optimizer_group.add_argument('--adam-private', action='store_const', const=tr.OptimizerType.ADAM_PRIVATE, dest='optimizer', help='Use Adam with private moments',
                                 default=tr.OptimizerType.ADAM_PRIVATE)
    optimizer_group.add_argument('--adam-shared', action='store_const', const=tr.OptimizerType.ADAM_SHARED, dest='optimizer', help='Use Adam with shared moments')
    optimizer_group.add_argument('--sgd', action='store_const', const=tr.OptimizerType.SGD, dest='optimizer', help='Use SGD')

    benchmark = sp.add_parser('benchmark', help='Benchmark train performance of an Ithemal setup')
    benchmark.add_argument('--n-examples', type=int, default=1000, help='Number of examples to use in benchmark')
    benchmark.add_argument('--trainers', type=int, default=4, help='Number of trainer processes to use')
    benchmark.add_argument('--threads', type=int,  default=4, help='Total number of PyTorch threads to create per trainer')
    benchmark.add_argument('--batch-size', type=int, default=4, help='The batch size to use in train')

    validate = sp.add_parser('validate', help='Get performance of a dataset')
    validate.add_argument('--load-file', help='File to load the model from')
    validate.add_argument('--iaca-only', help='Only report accuracy on IACA datapoints', action='store_true', default=False)

    dump = sp.add_parser('dump', help='Dump the dataset to a file')
    dump.add_argument('--dump-file', help='File to dump the model to', required=True)

    args = parser.parse_args()

    base_params = BaseParameters(
        data=args.data,
        embed_mode=args.embed_mode,
        embed_file=args.embed_file,
        random_edge_freq=args.random_edge_freq,
        predict_log=args.predict_log,
        no_residual=args.no_residual,
        no_dag_rnn=args.no_dag_rnn,
        dag_reduction=args.dag_reduction,
        edge_ablation_types=args.edge_ablations or [],
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        linear_embeddings=args.linear_embeddings,
        use_rnn=args.use_rnn,
        rnn_type=args.rnn_type,
        rnn_hierarchy_type=args.rnn_hierarchy_type,
        rnn_connect_tokens=args.rnn_connect_tokens,
        rnn_skip_connections=args.rnn_skip_connections,
        rnn_learn_init=args.rnn_learn_init,
        no_mem=args.no_mem,
        linear_dependencies=args.linear_dependencies,
        flat_dependencies=args.flat_dependencies,
        dag_nonlinearity=args.dag_nonlinearity,
        dag_nonlinearity_width=args.dag_nonlinearity_width,
        dag_nonlinear_before_max=args.dag_nonlinear_before_max,
    )

    if args.subparser == 'train':
        if args.split_dist:
            split = args.split_dist
        else:
            split = args.split_size or 1000

        train_params = TrainParameters(
            experiment_name=args.experiment_name,
            experiment_time=args.experiment_time,
            load_file=args.load_file,
            batch_size=args.batch_size,
            trainers=args.trainers,
            threads=args.threads,
            decay_trainers=args.decay_trainers,
            weight_decay=args.weight_decay,
            initial_lr=args.initial_lr,
            decay_lr=args.decay_lr,
            epochs=args.epochs,
            split=split,
            optimizer=args.optimizer,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weird_lr=args.weird_lr,
            lr_decay_rate=args.lr_decay_rate,
        )
        training.run_training_coordinator(base_params, train_params)

    elif args.subparser == 'validate':
        graph_model_validate(base_params, args.load_file, args.iaca_only)

    elif args.subparser == 'dump':
        graph_model_dump(base_params, args.dump_file)

    elif args.subparser == 'benchmark':
        benchmark_params = BenchmarkParameters(
            batch_size=args.batch_size,
            trainers=args.trainers,
            threads=args.threads,
            examples=args.n_examples,
        )
        graph_model_benchmark(base_params, benchmark_params)

    else:
        raise ValueError('Unknown mode "{}"'.format(args.subparser))

if __name__ == '__main__':
    main()
