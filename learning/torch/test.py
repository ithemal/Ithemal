import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.x = nn.Parameter(torch.zeros(1))
    
  
    def forward(self, x):

        return torch.mul(self.x,self.x) - self.x - 6


if __name__ == '__main__':

    model = Model()
    print(model)
    print list(model.parameters())
    print model.forward(torch.zeros(1))
    
    model.zero_grad()
    x = Variable(torch.zeros(1), requires_grad = True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    out = model(x)
    print out

    p = list(model.parameters())[0]

    loss = model.forward(x)
    loss.backward()

    while torch.abs(p.grad).data.numpy()[0] > 0.001: 
        optimizer.zero_grad()
        loss = model.forward(x)
        print p.grad.data.numpy(),p.data.numpy()
        loss.backward()
        optimizer.step()
        
        
