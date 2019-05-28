import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import torch
import torch.nn as nn
from enum import Enum
import common_libs.utilities as ut
import data.data_cost as dt
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np
import os
import gc
import psutil
from tqdm import tqdm
import time
import torch
from torch import nn
import utils.messages as messages
import random
from typing import Any, Callable, Dict, IO, List, Optional, Tuple

from . import model_utils

def memReport():
    # type: () -> None
    num_obj = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            num_obj += 1
    print 'num_obj ' + str(num_obj)

def cpuStats():
    # type: () -> None
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

class PredictionType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class OptimizerType(Enum):
    ADAM_PRIVATE = 1
    ADAM_SHARED = 2
    SGD = 3

class Train():

    """
    Performs training and validation for the models listed above
    """

    def __init__(self,
                 model,
                 data,
                 typ,
                 loss_fn,
                 num_losses,
                 batch_size = 1000,
                 tolerance = 25.,
                 lr = 0.001,
                 momentum = 0.9,
                 nesterov=False,
                 clip = 2.,
                 opt = OptimizerType.SGD,
                 weight_decay = 0.,
                 predict_log = False,
    ):
        # type: (nn.Module, dt.Data, PredictionType, Callable[[torch.tensor, torch.tensor], torch.tensor], int, int, float, float, float, bool, Optional[float], OptimizerType, float, bool) -> None

        self.model = model
        self.typ = typ
        self.data = data
        self.lr = lr
        self.clip = clip
        self.predict_log = predict_log
        self.opt_type = opt

        if opt == OptimizerType.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif opt == OptimizerType.ADAM_PRIVATE or opt == OptimizerType.ADAM_SHARED:
            self.optimizer = optim.Adam(self.model.parameters(), weight_decay=weight_decay, lr=lr)
            if opt == OptimizerType.ADAM_SHARED:
                for param in self.optimizer.param_groups[0]['params']:
                    param.share_memory_()
        else:
            raise ValueError('unknown optimizer...')

        #training parameters
        self.partition = (0, len(self.data.train))

        self.batch_size = batch_size

        #correctness
        self.tolerance = tolerance

        #for classification
        self.correct = 0

        #functions
        self.loss_fn = loss_fn
        self.num_losses = num_losses

        self.rank = 0
        self.last_save_time = 0

    def dump_shared_params(self):
        # type: () -> Dict[str, object]

        if self.opt_type == OptimizerType.ADAM_SHARED:
            return model_utils.dump_shared_params(self.optimizer)
        else:
            return {}

    def load_shared_params(self, params):
        # type: (Dict[str, object]) -> None

        if self.opt_type == OptimizerType.ADAM_SHARED:
            model_utils.load_shared_params(self.optimizer, params)

    """
    Print routines for predicted and target values.
    """
    def print_final(self,f,x,y):
        # type: (IO[str], np.array, np.array) -> None
        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    def print_max(self,f,x,y):
        # type: (IO[str], np.array, np.array) -> None
        x = torch.argmax(x)
        y = torch.argmax(y)

        f.write('%d,%d\n' % (x.item(),y.item()))

    """
    correct example counting functions
    """
    def correct_classification(self,x,y):
        # type: (torch.tensor, torch.tensor) -> None

        x = torch.argmax(x) + 1
        y = torch.argmax(y) + 1

        percentage = torch.abs(x - y) * 100.0 / y

        if percentage < self.tolerance:
            self.correct += 1

    def correct_regression(self,x,y):
        # type: (torch.tensor, torch.tensor) -> None

        if x.shape != ():
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    def save_checkpoint(self, epoch, batch_num, filename, **rest):
        # type: (int, int, str, **Any) -> None

        state_dict = {
            'epoch': epoch,
            'batch_num': batch_num,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        for (k, v) in rest.items():
            state_dict[k] = v

        # ensure directory exists
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError:
            pass

        torch.save(state_dict, filename)

    def load_checkpoint(self, filename):
        # type: (str) -> Dict[str, Any]

        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model'])

        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except ValueError:
            print('Couldnt load optimizer!')

        return state_dict

    def __call__(self, rank, partition, report_loss_fn=None):
        # type: (int, Tuple[int, int], Optional[Callable[[messages.Message], None]]) -> None
        self.rank = rank
        self.partition = partition
        self.train(report_loss_fn=report_loss_fn)

    def get_target(self, datum):
        # type: (dt.DataItem) -> torch.tensor
        target = torch.FloatTensor([datum.y]).squeeze()
        if self.predict_log:
            target.log_()
        return target

    """
    Training loop - to do make the average loss for general
    """

    def train(self, report_loss_fn=None):
        # type: (Optional[Callable[[messages.Message], None]]) -> None

        (partition_start, partition_end) = self.partition

        def report_trainer_death(idx):
            # type: (int) -> None

            if report_loss_fn is not None:
                report_loss_fn(messages.TrainerDeathMessage(
                    (idx + self.batch_size, partition_end),
                ))

        for idx in range(partition_start, partition_end, self.batch_size):
            batch_loss_sum = np.zeros(self.num_losses)
            self.correct = 0

            self.optimizer.zero_grad()
            loss_tensor = torch.FloatTensor([0]).squeeze()
            batch = self.data.train[idx:idx+self.batch_size]

            if not batch:
                continue

            for datum in batch:
                output = self.model(datum)

                if torch.isnan(output).any():
                    report_trainer_death(idx)
                    return

                #target as a tensor
                target = self.get_target(datum)

                #get the loss value
                if self.loss_fn:
                    losses_opt = self.loss_fn(output, target)

                if self.predict_log and self.loss_fn:
                    losses_rep = self.loss_fn(output.exp(), target.exp())
                else:
                    losses_rep = losses_opt

                #check how many are correct
                if self.typ == PredictionType.CLASSIFICATION:
                    self.correct_classification(output, target)
                elif self.typ == PredictionType.REGRESSION:
                    self.correct_regression(output, target)

                #accumulate the losses
                for class_idx, (loss_opt, loss_rep) in enumerate(zip(losses_opt, losses_rep)):
                    loss_tensor += loss_opt
                    l = loss_rep.item()
                    batch_loss_sum[class_idx] += l

            batch_loss_avg = batch_loss_sum / len(batch)

            #propagate gradients
            loss_tensor.backward()

            #clip the gradients
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

            for param in self.model.parameters():
                if param.grad is None:
                    continue

                if torch.isnan(param.grad).any():
                    report_trainer_death(idx)
                    return

            #optimizer step to update parameters
            self.optimizer.step()

            # get those tensors out of here!
            for datum in batch:
                self.model.remove_refs(datum)

            if report_loss_fn is not None:
                report_loss_fn(messages.LossReportMessage(
                    self.rank,
                    batch_loss_avg[0],
                    len(batch),
                ))

    def set_lr(self, lr):
        # type: (float) -> None
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    """
    Validation with a test set
    """

    def validate(self, resultfile, loadfile=None):
        # type: (str, Optional[str]) -> Tuple[List[List[float]], List[List[float]]]
        if loadfile is not None:
            print 'loaded from checkpoint for validation...'
            self.load_checkpoint(loadfile)

        f = open(resultfile,'w')

        self.correct = 0
        average_loss = [0] * self.num_losses
        actual = []
        predicted = []

        for j, item in enumerate(tqdm(self.data.test)):

            #print len(item.x)
            output = self.model(item)
            target = self.get_target(item)

            if self.predict_log:
                output.exp_()
                target.exp_()

            #get the target and predicted values into a list
            if self.typ == PredictionType.CLASSIFICATION:
                actual.append((torch.argmax(target) + 1).data.numpy().tolist())
                predicted.append((torch.argmax(output) + 1).data.numpy().tolist())
            else:
                actual.append(target.data.numpy().tolist())
                predicted.append(output.data.numpy().tolist())

            self.print_final(f, output, target)
            losses = self.loss_fn(output, target)
            if self.typ == PredictionType.CLASSIFICATION:
                self.correct_classification(output, target)
            else:
                self.correct_regression(output, target)

            #accumulate the losses
            loss = torch.zeros(1)
            for c,l in enumerate(losses):
                loss += l
                average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)

            if j % (len(self.data.test) / 100) == 0:
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                p_str += str(self.correct) + ' '
                print p_str

            #remove refs; so the gc remove unwanted tensors
            self.model.remove_refs(item)

        for loss in average_loss:
            f.write('loss - %f\n' % (loss))
        f.write('%f,%f\n' % (self.correct, len(self.data.test)))

        print average_loss, self.correct, len(self.data.test)
        f.close()

        return (actual, predicted)
