import torch
import torch.nn as nn
import sys
sys.path.append('..')
import common_libs.utilities as ut
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
from typing import Callable, Optional

def memReport():
    num_obj = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            num_obj += 1
    print 'num_obj ' + str(num_obj)

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


class Train():

    """
    Performs training and validation for the models listed above
    """

    def __init__(self,
                 model,
                 data,
                 batch_size = 1000,
                 tolerance = 25,
                 saves_per_epoch = 5,
                 lr = 0.001,
                 momentum = 0.9,
                 clip = 2,
                 opt = 'SGD',
                 weight_decay = None,
    ):
        self.model = model
        print self.model
        self.data = data
        self.lr = lr
        self.momentum = momentum
        self.clip = clip
        self.opt = opt
        self.weight_decay = weight_decay

        if opt == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif opt == 'Adam':
            if self.weight_decay:
                wd = self.weight_decay
            else:
                wd = 0
            self.optimizer = optim.Adam(self.model.parameters(), weight_decay=wd)
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
        self.loss_fn = None
        self.print_fn = None
        self.correct_fn = None
        self.num_losses = None

        #for plotting
        self.per_epoch_loss = []
        self.loss = []

        #training checkpointing
        self.saves_per_epoch = saves_per_epoch

        self.epoch_id = None
        self.rank = None
        self.last_save_time = 0

    """
    Print routines for predicted and target values.
    """
    def print_final(self,f,x,y):

        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    def print_max(self,f,x,y):

        x = torch.argmax(x)
        y = torch.argmax(y)

        f.write('%d,%d\n' % (x.item(),y.item()))

    """
    correct example counting functions
    """
    def correct_classification(self,x,y):

        x = torch.argmax(x) + 1
        y = torch.argmax(y) + 1

        percentage = torch.abs(x - y) * 100.0 / y

        if percentage < self.tolerance:
            self.correct += 1

    def correct_regression(self,x,y):

        if x.shape != ():
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    def save_checkpoint(self, epoch, batch_num, filename, **rest):

        state_dict = {
            'epoch': epoch,
            'batch_num': batch_num,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        for (k, v) in rest.items():
            state_dict[k] = v

        torch.save(state_dict, filename)

    def load_checkpoint(self, filename):

        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        return state_dict

    def __call__(self, epoch_id, rank, partition,
                 savefile=None, start_time=None, report_loss_fn=None):
        self.epoch_id = epoch_id
        self.rank = rank
        self.partition = partition
        self.train(savefile=savefile, start_time=start_time, report_loss_fn=report_loss_fn)

    """
    Training loop - to do make the average loss for general
    """

    def train(self, savefile=None, loadfile=None, start_time=None, report_loss_fn=None):
        # type: (Optional[str], Optional[str], Optional[float], Optional[Callable[[messages.LossReportMessage], None]]) -> None
        self.per_epoch_loss = []
        self.loss = []

        train_length = self.partition[1] - self.partition[0]
        epoch_len = train_length // self.batch_size
        leftover = train_length % self.batch_size

        epoch_sum_loss = np.zeros(self.num_losses)
        epoch_ema_loss = np.ones(self.num_losses)

        data = [self.data.train[i] for i in random.sample(range(*self.partition), train_length)]

        for batch_idx in range(epoch_len + 1):
            batch_loss_sum = np.zeros(self.num_losses)
            self.correct = 0

            self.optimizer.zero_grad()
            loss_tensor = torch.FloatTensor([0]).squeeze()

            batch_idx_start = batch_idx * self.batch_size
            batch = data[batch_idx_start:batch_idx_start+self.batch_size]

            if not batch:
                break

            for datum in batch:
                output = self.model(datum)

                if torch.isnan(output).any():
                    print 'output nan detected, quit learning, please use the saved model...'
                    #also add the per epch loss to the main loss accumulation
                    self.loss.append(self.per_epoch_loss)
                    return

                #target as a tensor
                target = torch.FloatTensor([datum.y]).squeeze()

                #get the loss value
                losses = self.loss_fn(output, target)

                #check how many are correct
                self.correct_fn(output, target)

                #accumulate the losses
                for class_idx, loss in enumerate(losses):
                    loss_tensor += loss
                    l = loss.item()
                    epoch_sum_loss[class_idx] += l
                    epoch_ema_loss[class_idx] = 0.98 * epoch_ema_loss[class_idx] + 0.02 * l
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

                isnan = torch.isnan(param.grad)
                if isnan.any():
                    print 'gradient values are nan...'
                    #append the loss before returning
                    self.loss.append(self.per_epoch_loss)
                    return

            #optimizer step to update parameters
            self.optimizer.step()

            # get those tensors out of here!
            for datum in batch:
                self.model.remove_refs(datum)

            #losses accumulation to visualize learning
            losses = []
            for av in epoch_sum_loss:
                losses.append(av / (batch_idx + 1))
            self.per_epoch_loss.append(losses)

            #change learning rates
            if self.correct_fn == self.correct_regression and self.opt != 'Adam':
                if epoch_sum_loss[0] / (batch_idx + 1) < 0.10 and self.lr > 0.00001:
                    print 'reducing learning rate....'
                    self.set_lr(self.lr * 0.1)

            if report_loss_fn is not None:
                report_loss_fn(messages.LossReportMessage(
                    self.rank,
                    batch_loss_avg[0],
                    len(batch),
                ))

        #loss accumulation
        self.loss.append(self.per_epoch_loss)
        self.per_epoch_loss = []
        self.set_lr(self.lr * 0.1)

    def set_lr(self, lr):
        # type: (float) -> None
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    """
    Validation with a test set
    """

    def validate(self, resultfile, loadfile=None):
        if loadfile != None:
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
            target = torch.FloatTensor([item.y]).squeeze()

            #get the target and predicted values into a list
            if self.correct_fn == self.correct_classification:
                actual.append((torch.argmax(target) + 1).data.numpy().tolist())
                predicted.append((torch.argmax(output) + 1).data.numpy().tolist())
            else:
                actual.append(target.data.numpy().tolist())
                predicted.append(output.data.numpy().tolist())

            self.print_fn(f, output, target)
            losses = self.loss_fn(output, target)
            self.correct_fn(output, target)

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
