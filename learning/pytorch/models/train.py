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
                 epochs = 3,
                 batch_size = 1000,
                 tolerance = 25,
                 epoch_len_div = 1,
                 saves_per_epoch = 5,
                 lr = 0.001,
                 momentum = 0.9,
                 clip = 2,
                 opt = 'SGD'
    ):
        self.model = model
        print self.model
        self.data = data
        self.lr = lr
        self.momentum = momentum
        self.clip = clip
        self.opt = opt
        if opt == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum)
        elif opt == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            print 'unknown optimizer...'
            exit(-1)

        #training parameters
        self.partition = (0, len(self.data.train))
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch_len_div = epoch_len_div

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

    def save_checkpoint(self, epoch, batch_num, filename):

        state_dict = {
            'epoch': epoch,
            'batch_num': batch_num,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, filename)

    def load_checkpoint(self, filename):

        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        epoch = state_dict['epoch']
        batch_num = state_dict['batch_num']

        return (epoch, batch_num)

    def __call__(self, id, partition) :
        self.partition = partition
        self.train()

    """
    Training loop - to do make the average loss for general
    """

    def train(self, savefile=None, loadfile=None):

        self.per_epoch_loss = []
        
        self.loss = []

        train_length = self.partition[1] - self.partition[0]

        # XXX: potentially drops self.batch_size - 1 elements
        # if train_length is not an even multiple
        epoch_len = (train_length // self.batch_size)
        epoch_len = epoch_len / self.epoch_len_div

        print 'start training...'
        print 'epochs ' + str(self.epochs)
        print "partition = (%d, %d)" % (self.partition)
        print 'epoch length ' + str(epoch_len)
        print 'batch size (sampled) ' + str(self.batch_size)

        restored_epoch = -1
        restored_batch_num = -1
        if loadfile != None:
            (restored_epoch, restored_batch_num) = self.load_checkpoint(loadfile)
            print 'starting from a checkpointed state... epoch %d batch_num %d' % (restored_epoch, restored_batch_num)


        for i in range(self.epochs):

            average_loss = [0] * self.num_losses          
            epoch_start = time.time();

            for j in range(epoch_len):
 
                start = time.time()

                if i <= restored_epoch and j <= restored_batch_num:
                    continue

                self.data.generate_batch(self.batch_size, self.partition)

                average_loss_per_batch = [0] * self.num_losses
                self.correct = 0
 
                #zero out grads
                self.optimizer.zero_grad()

                loss = torch.FloatTensor([0]).squeeze()
              

                #we use batches of size one for actual training
                for batch_j, item in enumerate(self.data.batch):
                    #get predicted value
                    output = self.model(item)

                    #check if output is nan, if so return
                    isnan = torch.isnan(output)

                    if isnan.any():
                        print 'output nan detected, quit learning, please use the saved model...'
                        #also add the per epch loss to the main loss accumulation
                        self.loss.append(self.per_epoch_loss)
                        return


                    #target as a tensor
                    target = torch.FloatTensor([item.y]).squeeze()

                    #get the loss value
                    losses = self.loss_fn(output, target)

                    #check how many are correct
                    self.correct_fn(output, target)

                    #accumulate the losses
                    for c,l in enumerate(losses):
                        item_num = j * self.batch_size + batch_j
                        average_loss[c] = (average_loss[c] * item_num + l.item()) / (item_num + 1)
                        average_loss_per_batch[c] = (average_loss_per_batch[c] * batch_j  + l.item()) / (batch_j + 1)

                    for loss_num in range(0,len(losses)):
                        loss = loss + losses[loss_num]



                #propagate gradients
                loss.backward()

                #clip the gradients
                if self.clip != None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                
                for param in self.model.parameters():
                    isnan = torch.isnan(param.grad)
                    if isnan.any():
                        print 'gradient values are nan...'
                        #append the loss before returning
                        self.loss.append(self.per_epoch_loss)
                        return

                #optimizer step to update parameters
                self.optimizer.step()

                # remove refs; so the gc remove unwanted tensors
                # self.model.remove_refs(item)
                  
                end = time.time()    

                if savefile != None:
                    self.save_checkpoint(i,j,savefile)

                #per batch training messages
                p_str = 'PID: %d ' % ( os.getpid(), ) + str(i) + ' ' + str(j) + ' '
                for av in average_loss:
                    p_str += "%.4f" % (av,) + ' '
                for av in average_loss_per_batch:
                    p_str += "%.4f" % (av,) + ' '
                p_str += str(self.correct) + ' ' + str(self.batch_size)
                p_str += " time: %.2f" % (end-start, ) 
                print p_str
                
                #losses accumulation to visualize learning
                losses = []
                for av in average_loss:
                    losses.append(av)
                self.per_epoch_loss.append(losses)

                #change learning rates
                if self.correct_fn == self.correct_regression and self.opt != 'Adam':
                    if average_loss_per_batch[0] < 0.10 and self.lr > 0.00001:
                        print 'reducing learning rate....'
                        self.lr = self.lr * 0.1
                        print 'learning rate changed ' + str(self.lr)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr
    
            epoch_end = time.time()
            print "Completed epoch %d  time: %s" % (i, epoch_end - epoch_start,)


            #loss accumulation
            self.loss.append(self.per_epoch_loss)
            self.per_epoch_loss = []

            #learning rate changes
            self.lr = self.lr * 0.1
            print 'learning rate changed ' + str(self.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


        if savefile != None:
            print 'final model saved...'
            self.save_checkpoint(self.epochs - 1,epoch_len - 1,savefile)


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








