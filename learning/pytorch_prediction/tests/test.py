import torch
import torch.nn as nn


def test_del():

    x = range(11)
    y = 2 * range(11)

    print x
    print y

    for i, val in enumerate(x):
        if val % 2 == 0:
            del x[i]
            del y[i]

    print x
    print y


def test_rankingloss():

    input1 = torch.ones(5)
    input2 = torch.ones(5)
    output = torch.ones(5)

    loss = nn.MarginRankingLoss()
    
    print input1
    print input2
    print output

    print loss(input1, input2, output)
    

def test_mseloss():

    loss = nn.MSELoss()
    loss1 = nn.MSELoss(size_average = False, reduce = True)
    loss2 = nn.MSELoss(reduce = False)
    input = torch.randn(3,5)
    target = torch.randn(3,5)
    
    print input, target

    output = loss(input, target)
    output1 = loss1(input, target)
    output2 = loss2(input, target)

    print output, output1, output2, torch.sum(output2).item(), torch.mean(output2).item()



if __name__ == "__main__":

    test_del()
