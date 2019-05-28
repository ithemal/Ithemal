#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import models.graph_models as md
import models.losses as ls
import models.train as tr
import data.data_cost as dt
from experiments.experiment import Experiment
import utils.messages as messages
from mpconfig import MPConfig
from utils import *
from training_messages import *
from ithemal_utils import *

import atexit
import collections
from enum import Enum
import time
import torch
from typing import Any, Dict, List, Iterator, Tuple, Type, Union, NamedTuple, TypeVar
import zmq
from tqdm import tqdm
import subprocess
import random
import uuid


# ------------------------------- TRAINER STATE ---------------------------------

class TrainerState(Enum):
    UNINITIALIZED = 0
    LOADING_DATA = 1
    READY_FOR_EPOCH = 2
    READY_FOR_DATA = 3
    DEAD = 4

# ------------------------------- LOSS REPORTING --------------------------------

class LossReporter(object):
    def __init__(self, experiment, n_datapoints, trainer):
        # type: (Experiment, int, tr.Train) -> None

        self.experiment = experiment
        self.n_datapoints = n_datapoints
        self.trainer = trainer

        self.start_time = time.time()
        self.ema_loss = 1.0
        self.running_trainers = 0
        self.epoch_no = 0
        self.total_processed_items = 0
        self.epoch_processed_items = 0

        self.last_report_time = 0.0
        self.last_save_time = 0.0

        self.root_path = experiment.experiment_root_path()

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        # line buffered
        self.loss_report_file = open(os.path.join(self.root_path, 'loss_report.log'), 'w', 1)

        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def format_loss(self):
        # type: () -> str

        return 'Epoch {}, Loss: {:.2}'.format(
            self.epoch_no,
            self.ema_loss,
        )

    def start_epoch(self, epoch_no, n_trainers):
        # type: (int, int) -> None

        self.epoch_no = epoch_no
        self.running_trainers = n_trainers
        self.epoch_processed_items = 0

        self.pbar.close()
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report_items(self, n_items, loss):
        # type: (int, float) -> None

        eps = 0.00025 * n_items

        self.ema_loss = self.ema_loss * (1 - eps) + loss * eps
        self.epoch_processed_items += n_items
        self.total_processed_items += n_items

        desc = self.format_loss()
        self.pbar.set_description(desc)
        self.pbar.update(n_items)

    def report_trainer_death(self):
        # type: () -> None

        self.running_trainers -= 1
        self.pbar.write('Trainer died! Down to {} trainers'.format(self.running_trainers))

    def _report_loss(self, t):
        # type: (float) -> None

        message = '\t'.join(map(str, (
            self.epoch_no,
            t - self.start_time,
            self.ema_loss,
            self.running_trainers
        )))
        self.loss_report_file.write(message + '\n')

    def _checkpoint_trainer(self, t):
        # type: (float) -> None

        checkpoint_fname = self.experiment.checkpoint_file_name(t - self.start_time)
        self.trainer.save_checkpoint(
            self.epoch_no, 0, checkpoint_fname,
            runtime=t - self.start_time,
            ep_proc_instances=self.epoch_processed_items,
            total_proc_instances=self.total_processed_items,
        )

    def report(self):
        # type: () -> None

        t = time.time()
        if t - self.last_report_time > 10:
            self._report_loss(t)
            self.last_report_time = t

        if t - self.last_save_time > 10*60:
            self._checkpoint_trainer(t)
            self.last_save_time = t

    def finish(self):
        # type: () -> None

        self.pbar.close()
        print('Finishing training')

        t = time.time()
        self._report_loss(t)
        self._checkpoint_trainer(t)

        self.trainer.save_checkpoint(
            self.epoch_no,
            0,
            os.path.join(self.root_path, 'trained.mdl')
        )

        resultfile = os.path.join(self.root_path, 'validation_results.txt')
        self.trainer.validate(resultfile)

# ------------------------------ DATA PARTITIONING ------------------------------
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

def get_partitions(n_datapoints, train_params):
    # type: (int, TrainParameters) -> List[Tuple[int, int]]

    split = train_params.split

    if isinstance(split, int):
        return list(get_partition_splits_from_size(n_datapoints, split))
    else:
        return list(get_partition_splits_from_distr(n_datapoints, train_params.trainers, split))


# ---------------------------- TRAINER CONSTRUCTION -----------------------------

def load_trainer(base_params, train_params, model, data):
    # type: (BaseParameters, TrainParameters, md.AbstractGraphModule, dt.DataCost) -> tr.Train

    return tr.Train(
        model, data, tr.PredictionType.REGRESSION, ls.mse_loss, 1,
        batch_size=train_params.batch_size, clip=None, opt=train_params.optimizer,
        lr=train_params.initial_lr, weight_decay=train_params.weight_decay,
        predict_log=base_params.predict_log, momentum=train_params.momentum,
        nesterov=train_params.nesterov,
    )

# -------------------------------- COORDINATION ---------------------------------

def get_socket_url(identifier):
    # type: (str) -> str

    return 'ipc:///tmp/{}.socket'.format(identifier)

def run_training_coordinator(base_params, train_params):
    # type: (BaseParameters, TrainParameters) -> None

    torch.multiprocessing.set_sharing_strategy('file_system')
    expt = Experiment(train_params.experiment_name, train_params.experiment_time, base_params.data)

    socket_identifier = str(uuid.uuid4())

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(get_socket_url(socket_identifier))

    def send_msg(msg):
        # type: (Union[object, List[object]]) -> None
        if isinstance(msg, list):
            socket.send_pyobj(msg)
        else:
            socket.send_pyobj([msg])

    # fork off trainers
    procs = []
    mp_config = MPConfig(train_params.threads)
    trainer_states = {} # type: Dict[int, TrainerState]

    def all_in_state(state):
        # type: (TrainerState) -> bool
        return all(trainer_states[rank] == state for rank in trainer_states)

    with mp_config:
        for idx in range(train_params.trainers):
            trainer_states[idx] = TrainerState.UNINITIALIZED
            mp_config.set_env(idx)
            procs.append(subprocess.Popen([sys.executable, __file__, socket_identifier, str(idx)]))

    @atexit.register
    def cleanup_procs():
        # type: () -> None
        print('cleaning up trainers')
        for proc in procs:
            proc.terminate()

    while not all_in_state(TrainerState.LOADING_DATA):
        msg = socket.recv_pyobj()
        if isinstance(msg, TrainerInitializeReq):
            send_msg(TrainerInitializeResp(
                base_params,
                train_params,
            ))
            trainer_states[msg.rank] = TrainerState.LOADING_DATA
        elif isinstance(msg, TrainerDataReq):
            send_msg(WaitResp())
        else:
            raise ValueError('Unexpected message {}'.format(msg))

    data = load_data(base_params)
    model = load_model(base_params, data)

    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    trainer = load_trainer(base_params, train_params, model, data)

    while not all_in_state(TrainerState.READY_FOR_EPOCH):
        msg = socket.recv_pyobj()
        if isinstance(msg, TrainerDataReq):
            send_msg(TrainerDataResp(
                model.dump_shared_params(),
                trainer.dump_shared_params(),
            ))
            trainer_states[msg.rank] = TrainerState.READY_FOR_EPOCH
        elif isinstance(msg, TrainerStepReq):
            send_msg(WaitResp())
        else:
            raise ValueError('Unexpected message {}'.format(msg))

    current_lr = train_params.initial_lr
    loss_reporter = LossReporter(expt, len(data.train), trainer)

    for epoch_no in range(train_params.epochs):
        if train_params.decay_trainers:
            n_trainers = max(1, train_params.trainers - epoch_no)
        else:
            n_trainers = train_params.trainers

        loss_reporter.start_epoch(epoch_no + 1, n_trainers)

        # start exactly n_trainers trainers, kill the rest
        n_started_trainers = 0
        while not all_in_state(TrainerState.READY_FOR_DATA):
            msg = socket.recv_pyobj()

            if isinstance(msg, TrainerStepReq):
                if trainer_states[msg.rank] == TrainerState.READY_FOR_EPOCH:
                    if n_started_trainers >= n_trainers:
                        send_msg(KillResp())
                        del trainer_states[msg.rank]
                    else:
                        send_msg([ShuffleDataResp(random.getstate()), SetLrResp(current_lr)])
                        trainer_states[msg.rank] = TrainerState.READY_FOR_DATA
                        n_started_trainers += 1
                else:
                    send_msg([WaitResp()])
            else:
                raise ValueError('Unexpected message {}'.format(msg))

        # shuffle data locally to permute random state
        random.shuffle(data.train)

        # get partitions
        partitions = get_partitions(len(data.train), train_params)
        partition_idx = 0

        # run until all done with epoch or dead
        while not all(trainer_states[rank] in (TrainerState.READY_FOR_EPOCH, TrainerState.DEAD) for rank in trainer_states):
            msg = socket.recv_pyobj()

            if trainer_states[msg.rank] == TrainerState.DEAD:
                send_msg(WaitResp())
            elif isinstance(msg, TrainerStepReq):
                if partition_idx < len(partitions):
                    trainer_states[msg.rank] = TrainerState.READY_FOR_DATA
                    send_msg(RunTrainerResp(partitions[partition_idx]))
                    partition_idx += 1
                else:
                    send_msg(WaitResp())
                    trainer_states[msg.rank] = TrainerState.READY_FOR_EPOCH
            elif isinstance(msg, TrainerLossReq):
                send_msg(TrainerLossResp())
                loss_reporter.report_items(msg.n_items, msg.loss)
            elif isinstance(msg, TrainerDeathReq):
                send_msg(TrainerDeathResp())
                loss_reporter.report_trainer_death()
                trainer_states[msg.rank] = TrainerState.DEAD
                if msg.partition_remainder[0] < msg.partition_remainder[1]:
                    partitions.append(msg.partition_remainder)
            else:
                raise ValueError('Unexpected Message {}'.format(msg))

            loss_reporter.report()

        if all_in_state(TrainerState.DEAD):
            break

        # reset states
        for rank in trainer_states:
            trainer_states[rank] = TrainerState.READY_FOR_EPOCH

        # decay LR if necessary
        if train_params.decay_lr or (train_params.weird_lr and epoch_no > 0):
            current_lr /= train_params.lr_decay_rate

    loss_reporter.finish()

# ----------------------------------- WORKER ------------------------------------

def run_training_worker(identifier, rank):
    # type: (str, int) -> None

    print('creating socket...')
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print('connecting to coordinator')
    socket.connect(get_socket_url(identifier))

    def send_msg(msg):
        # type: (Any) -> None
        socket.send_pyobj(msg)

    def recv_msgs():
        # type: () -> List[Any]
        return socket.recv_pyobj()

    def recv_one():
        # type: () -> Any
        resp = recv_msgs()
        assert len(resp) == 1
        return resp[0]

    T = TypeVar('T')
    def send_and_get_one(msg, typ):
        # type: (Any, Type[T]) -> T
        send_msg(msg)
        res = recv_one()
        assert isinstance(res, typ)
        return res

    initialize_params = send_and_get_one(TrainerInitializeReq(rank), TrainerInitializeResp)
    base_params = initialize_params.base_params
    train_params = initialize_params.train_params

    data = load_data(base_params)
    model = load_model(base_params, data)
    trainer = load_trainer(base_params, train_params, model, data)

    data_params = None
    while not isinstance(data_params, TrainerDataResp):
        send_msg(TrainerDataReq(rank))
        data_params = recv_one()

    model.load_shared_params(data_params.model_tensor_params)
    trainer.load_shared_params(data_params.trainer_tensor_params)

    loss_report_freq = 10
    losses = [] # type: List[Tuple[float, int]]

    def report_loss(msg):
        # type: (messages.Message) -> None
        if isinstance(msg, messages.TrainerDeathMessage):
            send_and_get_one(TrainerDeathReq(rank, msg.remaining_partition), TrainerDeathResp)
        elif isinstance(msg, messages.LossReportMessage):
            losses.append((msg.loss, msg.n_items))
            if len(losses) > loss_report_freq:
                avg_loss = sum(l[0] for l in losses) / len(losses)
                n_items = sum(l[1] for l in losses)
                losses[:] = []

                send_and_get_one(TrainerLossReq(rank, avg_loss, n_items), TrainerLossResp)
        else:
            raise ValueError('Unexpected message {}'.format(msg))

    print('starting train loop')
    while True:
        send_msg(TrainerStepReq(rank))
        msgs = recv_msgs()

        for msg in msgs:
            if isinstance(msg, WaitResp):
                time.sleep(1)
                continue
            elif isinstance(msg, KillResp):
                print('Trainer {} dying'.format(rank))
                return
            elif isinstance(msg, ShuffleDataResp):
                random.setstate(msg.random_state)
                random.shuffle(data.train)
                random.seed()
                continue
            elif isinstance(msg, SetLrResp):
                trainer.set_lr(msg.new_lr)
                continue
            elif isinstance(msg, RunTrainerResp):
                trainer(rank, msg.partition, report_loss)
                continue
            else:
                raise ValueError('Unexpected message {}'.format(msg))

def main():
    # type: () -> None

    assert len(sys.argv) == 3, 'Must be passed exactly two parameters: socket ID, rank'
    run_training_worker(sys.argv[1], int(sys.argv[2]))

if __name__ == '__main__':
    main()
