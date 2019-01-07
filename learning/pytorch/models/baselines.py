import torch
import torch.nn as nn
import random

class MeanPredictor(nn.Module):
    def __init__(self, data):
        super(MeanPredictor, self).__init__()
        self._throwaway_param = nn.Linear(1, 1)
        self.mean = sum(datum.y for datum in data.train) / float(len(data.train))
        print('we gonna predict {}'.format(self.mean))

    def remove_refs(self, arg):
        pass

    def forward(self, datum):
        return torch.tensor([self.mean]).squeeze()

class RandomPredictor(nn.Module):
    def __init__(self, data):
        super(RandomPredictor, self).__init__()
        self._throwaway_param = nn.Linear(1, 1)
        self.data = data

    def remove_refs(self, arg):
        pass

    def forward(self, datum):
        guess = random.choice(self.data.train).y
        return torch.tensor([guess]).squeeze()
