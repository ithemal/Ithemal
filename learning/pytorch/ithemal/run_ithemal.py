import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))
from enum import Enum
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

from utils import messages
import models.graph_models as md
import data.data_cost as dt
import models.losses as ls
import models.train as tr
from tqdm import tqdm
from mpconfig import MPConfig
from typing import Callable, List, Optional, Iterator, Tuple, NamedTuple, Union
from experiments import experiment
import random

class EdgeAblationType(Enum):
    TRANSITIVE_REDUCTION = 1
    TRANSITIVE_CLOSURE = 2
    ADD_LINEAR_EDGES = 3
    ONLY_LINEAR_EDGES = 4
    NO_EDGES = 5

BaseParameters = NamedTuple('BaseParameters', [
    ('data', str),
    ('embed_mode', str),
    ('embed_file', str),
    ('random_edge_freq', float),
    ('predict_log', bool),
    ('no_residual', bool),
    ('edge_ablation_type', Optional[EdgeAblationType]),
    ('embed_size', int),
    ('hidden_size', int),
])

TrainParameters = NamedTuple('TrainParameters', [
    ('experiment_name', str),
    ('experiment_time', str),
    ('load_file', Optional[str]),
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('decay_trainers', bool),
    ('weight_decay', float),
    ('initial_lr', float),
    ('decay_lr', bool),
    ('epochs', int),
    ('split', Union[int, List[float]]),
    ('optimizer', tr.OptimizerType),
])

BenchmarkParameters = NamedTuple('BenchmarkParameters', [
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('examples', int),
])


def ablate_data(data, edge_ablation_type, random_edge_freq):
    # type: (dt.DataCost, Optional[EdgeAblationType], float) -> None

    if edge_ablation_type == EdgeAblationType.TRANSITIVE_REDUCTION:
        for data_item in data.data:
            data_item.block.transitive_reduction()
    elif edge_ablation_type == EdgeAblationType.TRANSITIVE_CLOSURE:
        for data_item in data.data:
            data_item.block.transitive_closure()
    elif edge_ablation_type == EdgeAblationType.ADD_LINEAR_EDGES:
        for data_item in data.data:
            data_item.block.linearize_edges()
    elif edge_ablation_type == EdgeAblationType.ONLY_LINEAR_EDGES:
        for data_item in data.data:
            data_item.block.remove_edges()
            data_item.block.linearize_edges()
    elif edge_ablation_type == EdgeAblationType.NO_EDGES:
        for data_item in data.data:
            data_item.block.remove_edges()

    if random_edge_freq > 0:
        for data_item in data.data:
            data_item.block.random_forward_edges(random_edge_freq / len(data_item.block.instrs))

def load_model_and_data(params):
    # type: (BaseParameters) -> Tuple[md.GraphNN, dt.DataCost]
    data = dt.load_dataset(params.embed_file, data_savefile=params.data)
    ablate_data(data, params.edge_ablation_type, params.random_edge_freq)

    model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1, use_residual=not params.no_residual)
    model.set_learnable_embedding(mode=params.embed_mode, dictsize=max(data.word2id) + 1, seed=data.final_embeddings)

    return model, data

def get_partition_splits_from_distr(n_datapoints, n_trainers, split_distr):
    # type: (int, int, List[float]) -> Iterator[Tuple[int, int]]

    assert abs(sum(split_distr) - 1) < 1e-4
    assert all(elem >= 0 for elem in split_distr)

    idx = 0
    for frac in split_distr:
        split_size = int((n_datapoints / n_trainers) * frac)
        for tr in range(n_trainers):
            yield (idx, idx + split_size)
            idx += split_size
    yield (idx, n_datapoints)

def get_partition_splits_from_size(n_datapoints, split_size):
    # type: (int, int) -> Iterator[Tuple[int, int]]

    for i in range(0, n_datapoints, split_size):
        yield (i, i + split_size)

def graph_model_learning(base_params, train_params):
    # type: (BaseParameters, TrainParameters) -> None

    expt = experiment.Experiment(train_params.experiment_name, train_params.experiment_time, base_params.data)
    expt_root = expt.experiment_root_path()

    try:
        os.makedirs(expt_root)
    except OSError:
        pass

    model, data = load_model_and_data(base_params)

    lr = train_params.initial_lr
    train = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=train_params.batch_size, clip=None, opt=train_params.optimizer, lr=lr,
        weight_decay=train_params.weight_decay, predict_log=base_params.predict_log
    )

    #defining losses, correctness and printing functions
    restored_epoch = -1
    restored_batch_num = -1

    if train_params.load_file:
        state_dict = train.load_checkpoint(train_params.load_file)
        restored_epoch = state_dict['epoch']
        restored_batch_num = state_dict['batch_num']
        print('starting from a checkpointed state... epoch {} batch_num {}'.format(
            restored_epoch,
            restored_batch_num,
        ))

    model.share_memory()

    start_time = time.time()

    def loss_report_func(message_q, partition_q):
        # type: (mp.Queue, mp.Queue) -> None
        loss_report_file = open(os.path.join(expt_root, 'loss_report.log'), 'w', 1) # line buffered
        last_report_time = time.time()
        last_save_time = 0.

        def format_loss(ep_no, loss):
            # type: (Union[float, int, str], Union[float, str]) -> str
            return 'Epoch {}, Loss {:.2}'.format(ep_no, loss)

        pbar = tqdm(desc=format_loss('??', '??'), total=len(data.train))
        ema_loss = 1.0
        ep_no = 0
        running_trainers = 0
        ep_proc_instances = 0
        total_proc_instances = 0

        while True:
            item = message_q.get()
            if item is None:
                return
            elif isinstance(item, messages.LossReportMessage):
                loss = item.loss
                ema_loss = ema_loss * 0.999 + loss * 0.001
                ep_proc_instances += item.n_items
                total_proc_instances += item.n_items
                pbar.update(item.n_items)

            elif isinstance(item, messages.EpochAdvanceMessage):
                ep_proc_instances = 0
                ep_no = item.epoch
                running_trainers = item.n_trainers
                pbar.close()
                pbar = tqdm(desc=format_loss('??', '??'), total=len(data.train))

            elif isinstance(item, messages.TrainerDeathMessage):
                running_trainers -= 1
                pbar.write('Trainer died! Down to {} trainers'.format(running_trainers))
                remaining_part = item.remaining_partition
                if remaining_part[0] < remaining_part[1]:
                    partition_queue.put(remaining_part)

                if running_trainers == 0:
                    pbar.write('All trainers dead! Terminating learning')
                    try:
                        while True:
                            partition_queue.task_done()
                    except ValueError:
                        pass

            t = time.time()
            if t - last_report_time > 10:
                message = '\t'.join(map(str, (
                    ep_no,
                    t - start_time,
                    ema_loss,
                    running_trainers
                )))
                loss_report_file.write(message + '\n')
                last_report_time = t

            if t - last_save_time > 10*60:
                checkpoint_fname = expt.checkpoint_file_name(t - start_time)
                train.save_checkpoint(
                    ep_no, 0, checkpoint_fname,
                    runtime=t - start_time,
                    ep_proc_instances=ep_proc_instances,
                    total_proc_instances=total_proc_instances,
                )
                last_save_time = t

            pbar.set_description(format_loss(ep_no, ema_loss))

    def run_training(part_q, loss_report_func):
        # type: (mp.Queue, Callable[[messages.Message], None]) -> None
        while True:
            part = part_q.get()

            if p is None:
                part_q.task_done()
                return

            train(rank, part, loss_report_func)
            part_q.task_done()

    message_q = mp.Queue()
    partition_queue = mp.JoinableQueue()

    loss_proc = mp.Process(target=loss_report_func, args=(message_q, partition_queue))
    loss_proc.daemon = True
    loss_proc.start()

    for i in range(train_params.epochs):
        i = restored_epoch + i + 1
        if train_params.decay_trainers:
            n_trainers = max(1, train_params.trainers - i)
        else:
            n_trainers = train_params.trainers

        message_q.put(messages.EpochAdvanceMessage(i + 1, n_trainers))

        # shuffle the data before forking processes
        random.shuffle(data.train)

        mp_config = MPConfig(n_trainers, train_params.threads)
        processes = []
        with mp_config:
            for rank in range(mp_config.trainers):
                mp_config.set_env(rank)

                p = mp.Process(target=run_training, args=(partition_queue, message_q.put))
                p.daemon = True
                p.start()
                print("Starting process %d" % (rank,))
                processes.append(p)

        if isinstance(train_params.split, int):
            partitions = list(get_partition_splits_from_size(len(data.train), train_params.split))
        else:
            partitions = list(get_partition_splits_from_distr(len(data.train), mp_config.trainers, train_params.split))

        for partition in partitions:
            partition_queue.put(partition)

        # wait until all partitions processed
        partition_queue.join()

        # kill all trainer procs
        for p in processes:
            partition_queue.put(None)

        # wait for all trainer procs to finish
        for p in processes:
            p.join()

        train.save_checkpoint(i, 0, os.path.join(expt_root, 'trained.mdl'))

        # decay LR if necessary
        if train_params.decay_lr:
            lr /= 10
            train.set_lr(lr)

    message_q.put(None)

    resultfile = os.path.join(expt_root, 'validation_results.txt')
    results = train.validate(resultfile)

def graph_model_benchmark(base_params, benchmark_params):
    # type: (BaseParameters, BenchmarkParameters) -> None
    model, data = load_model_and_data(base_params)
    train = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=benchmark_params.batch_size, clip=None, opt=tr.OptimizerType.ADAM_PRIVATE, lr=0.01,
    )

    model.share_memory()

    mp_config = MPConfig(benchmark_params.trainers, benchmark_params.threads)
    partition_size = benchmark_params.examples // benchmark_params.trainers

    processes = []

    start_time = time.time()

    with mp_config:
        for rank in range(mp_config.trainers):
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

def graph_model_validate(base_params, model_file):
    # type: (BaseParameters, str) -> None
    model, data = load_model_and_data(base_params)

    train = tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=1000, clip=None, predict_log=base_params.predict_log,
    )

    #train.data.test = train.data.test[:10000]

    resultfile = os.environ['ITHEMAL_HOME'] + '/learning/pytorch/results/realtime_results.txt'
    (actual, predicted) = train.validate(resultfile=resultfile, loadfile=model_file)

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

    # edge/misc arguments
    parser.add_argument('--random-edge-freq', type=float, default=0.0, help='The fraction of instructions to add an additional random forward edge to (can be >1)')
    parser.add_argument('--no-residual', default=False, action='store_true', help='Don\'t use a residual model in Ithemal')
    parser.add_argument('--predict-log', action='store_true', default=False, help='Predict the log of the time')

    edge_ablation_parser_group = parser.add_mutually_exclusive_group()
    edge_ablation_parser_group.add_argument('--transitive-reduction', action='store_const', dest='edge_ablation', const=EdgeAblationType.TRANSITIVE_REDUCTION)
    edge_ablation_parser_group.add_argument('--transitive-closure', action='store_const', dest='edge_ablation', const=EdgeAblationType.TRANSITIVE_CLOSURE)
    edge_ablation_parser_group.add_argument('--add-linear-edges', action='store_const', dest='edge_ablation', const=EdgeAblationType.ADD_LINEAR_EDGES)
    edge_ablation_parser_group.add_argument('--only-linear-edges', action='store_const', dest='edge_ablation', const=EdgeAblationType.ONLY_LINEAR_EDGES)
    edge_ablation_parser_group.add_argument('--no-edges', action='store_const', dest='edge_ablation', const=EdgeAblationType.NO_EDGES)

    sp = parser.add_subparsers(dest='subparser')

    train = sp.add_parser('train', help='Train an ithemal model')
    train.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    train.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    train.add_argument('--load-file', help='Start by loading the provided model')

    train.add_argument('--batch-size', type=int, default=4, help='The batch size to use in train')
    train.add_argument('--epochs', type=int, default=5, help='Number of epochs to run for')
    train.add_argument('--trainers', type=int, default=4, help='Number of trainer processes to use')
    train.add_argument('--threads', type=int,  default=4, help='Total number of PyTorch threads to create per trainer')
    train.add_argument('--decay-trainers', action='store_true', default=False, help='Decay the number of trainers at the end of each epoch')
    train.add_argument('--weight-decay', type=float, default=0, help='Coefficient of weight decay (L2 regularization) on model')
    train.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate')
    train.add_argument('--decay-lr', action='store_true', default=False, help='Decay the learning rate at the end of each epoch')

    split_group = train.add_mutually_exclusive_group()
    split_group.add_argument(
        '--split-dist', nargs='+', type=float, default=[0.5, 0.25, 0.125, .0625, .0625],
        help='Split data partitions between trainers via a distribution',
        dest='split',
    )
    split_group.add_argument('--split-size', type=int, help='Partitions of a fixed size', dest='split')

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

    args = parser.parse_args()

    base_params = BaseParameters(
        data=args.data,
        embed_mode=args.embed_mode,
        embed_file=args.embed_file,
        random_edge_freq=args.random_edge_freq,
        predict_log=args.predict_log,
        no_residual=args.no_residual,
        edge_ablation_type=args.edge_ablation,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
    )

    if args.subparser == 'train':
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
            split=args.split,
            optimizer=args.optimizer,
        )
        graph_model_learning(base_params, train_params)

    elif args.subparser == 'validate':
        graph_model_validate(base_params, args.load_file)

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
