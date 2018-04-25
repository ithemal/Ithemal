import torch
import torch.nn as nn


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

    test_mseloss()
