import torch
import torch.nn as nn
import sys
sys.path.append('..')
import common.utilities as ut
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np


class Train(): 

    """
    Performs training and validation for the models listed above
    """

    def __init__(self, 
                 model,
                 data,
                 epochs = 3,
                 batch_size = 1000,
                 tolerance = 25):

        self.model = model
        print self.model
        self.data = data
        self.optimizer = optim.Adam(self.model.parameters())
        #self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)

        #training parameters
        self.epochs = epochs
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
        
    """
    Print routines for predicted and target values.
    """
    def print_final(self,f,x,y):
        
        size = x.shape[0]
        for i in range(size):
            f.write('%f,%f ' % (x[i],y[i]))
        f.write('\n')
    

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
        
        if x.shape[0] > 1:
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    
    """
    Training loop - to do make the average loss for general
    """

    def train(self):

        epoch_len = (len(self.data.train) // self.batch_size)

        print 'start training...'
        print 'epochs ' + str(self.epochs)
        print 'epoch length ' + str(epoch_len)
        print 'batch size (sampled) ' + str(self.batch_size)

        for i in range(self.epochs):
           
            average_loss = [0] * self.num_losses
            j = 0
            
            for _ in range(epoch_len):
 
                batch = self.data.generate_batch(self.batch_size)

                average_loss_per_batch = [0] * self.num_losses
                batch_j = 0
                self.correct = 0

                #we use batches of size one for actual training
                for item in batch:
                    
                    #zero out grads
                    self.optimizer.zero_grad()
  
                    #get predicted value
                    output = self.model(item)

                    #target as a tensor
                    target = torch.FloatTensor([item.y]).squeeze()
              
                    #get the loss value
                    losses = self.loss_fn(output, target)
                    
                    if batch_j == self.batch_size / 2:
                        self.print_fn(sys.stdout, output, target)

                    #check how many are correct
                    self.correct_fn(output, target)
                        
                    #accumulate the losses
                    loss = torch.zeros(1)
                    for c,l in enumerate(losses):
                        loss += l
                        average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)
                        average_loss_per_batch[c] = (average_loss_per_batch[c] * batch_j  + l.item()) / (batch_j + 1)

                    #propagate gradients
                    loss.backward()
                    
                    #optimizer step to update parameters
                    self.optimizer.step()

                    j += 1
                    batch_j += 1
                                        
                
                #per batch training messages
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                for av in average_loss_per_batch:
                    p_str += str(av) + ' '
                p_str += str(self.correct) + ' ' + str(self.batch_size)
                print p_str
                
           
            print i
    
        
    """
    Validation with a test set
    """

    def validate(self, filename):
        
        average_loss = [0] * self.num_losses
        j = 0

        f = open(filename,'w')

        print len(self.data.test)

        self.correct = 0

        for item in self.data.test:
            
            output = self.model(item)
            target = torch.FloatTensor([item.y]).squeeze()

            self.print_fn(f, output, target)

            losses = self.loss_fn(output, target)

            self.correct_fn(output, target)

            #accumulate the losses
            loss = torch.zeros(1)
            for c,l in enumerate(losses):
                loss += l
                average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)
            
            j += 1
            if j % (len(self.data.test) / 100) == 0:
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                p_str += str(self.correct) + ' '
                print p_str
           
        print average_loss, self.correct, len(self.data.test)
        f.close()            



        
        
        
