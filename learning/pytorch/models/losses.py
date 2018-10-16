import torch
import torch.nn as nn
import sys
sys.path.append('..')
import common_libs.utilities as ut
import torch.autograd as autograd
import torch.optim as optim
import math
import numpy as np

"""
mse loss normalized
"""
def mse_loss(output,target):


    loss_fn = nn.MSELoss(reduce = False)
    loss = torch.sqrt(loss_fn(output, target)) / (target + 1e-3)
    loss = torch.mean(loss)

    return [loss]

"""
mse loss + margin rank loss
"""

def mse_loss_plus_rank_loss(output,target):

    cost = output
    target_cost = target

    if output.size()[0] > 1:
        inter = output[:-1]
        inter_1 = output[1:]
    else: #emulate no rank loss
        inter = torch.ones(1)
        inter_1 = 2 * torch.ones(1)

    target_rank = torch.ones(inter.size())

    loss_mse = nn.MSELoss(reduce = False)
    loss1 = torch.sqrt(loss_mse(cost, target_cost)) / (target_cost + 1e-3)
    loss1 = torch.mean(loss1)

    loss_rank = nn.MarginRankingLoss()
    loss2 = loss_rank(inter_1, inter, target_rank)

    return [loss1, loss2]


"""
softmax cross entropy loss with weights
"""
def cross_entropy_loss_with_weights(output,target):

    outputs = nn.functional.softmax(output,0)

    nz = torch.nonzero(target)[0][0]

    mean = nz
    std = nz * 0.05
    weight_points = range(target.shape[0])

    normal = torch.distributions.normal.Normal(mean, std)
    weight_values = torch.exp(normal.log_prob(torch.FloatTensor(weight_points)))
    weight_values = weight_values / torch.sum(weight_values)

    target = weight_values * 100  #just scaling the weights

    loss = nn.functional.binary_cross_entropy(outputs,target)
    return [loss]


"""
softmax cross entropy loss for classification
"""
def cross_entropy_loss(output,target):

    outputs = nn.functional.softmax(output,0)

    loss = nn.functional.binary_cross_entropy(outputs,target)
    return [loss]

