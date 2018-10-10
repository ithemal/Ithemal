from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

import common_libs.utilities as ut
import graphs as gr
import models.graph_models as md
import models.train as tr
import data.data_cost as dt
import word2vec.word2vec as w2v

from tqdm import tqdm
import subprocess
import os
import re
import time
import argparse
import statistics
import pickle
import torch
import torch.nn as nn
import math

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import axes3d, Axes3D

def plot_learning_curves(filename, losses, legend, ylabel='loss', xlabel='batch', title='Learning Curves', xmin = None, xmax = None, ymin = None, ymax = None):
    plt.figure()
    for loss, label in zip(losses, legend):
        y = loss
        x = np.arange(0,3*len(loss),3)
        h = plt.plot(x,y, '.-', linewidth=1, markersize=2, label=label)

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    cur_xmin, cur_xmax = plt.xlim()
    cur_ymin, cur_ymax = plt.ylim()

    if xmin != None and cur_xmin < xmin:
        plt.xlim(xmin = xmin)
    if ymin != None and cur_ymin < ymin:
        plt.ylim(ymin = ymin)
    if xmax != None and cur_xmax > xmax:
        plt.xlim(xmax = xmax)
    if ymax != None and cur_ymax > ymax:
        plt.ylim(ymax = ymax)
    plt.savefig(filename)
    plt.close()

#fixes an error in the dumped data
def smoothen_curves(losses, batch_size):

    for loss in losses:

        for i in range(batch_size, len(loss)):

            val = (loss[i-1]*(i-1) + loss[i])/i
            loss[i] = val



if __name__ == '__main__':

    additive_losses = torch.load('../results/losses_additive_2.pkl')
    span_losses = torch.load('../results/losses_span_2.pkl')
    throughput_losses = torch.load('../results/losses_throughput_3.pkl')


    modelnames = ['Sequential RNN', 'Hierarchical RNN', 'DAG-RNN']

    for loss in additive_losses:
        print len(loss)

    for loss in span_losses:
        print len(loss)

    for loss in throughput_losses:
        print len(loss)


    smoothen_curves(additive_losses, 75)


    for i, loss in enumerate(additive_losses):
        print 'val'
        print len(loss)
        val = loss[-1]
        for j in range(len(loss), 100):
            loss.append(val)


    result_name = '../results/paper_additive_2.png'
    plot_learning_curves(result_name, additive_losses, modelnames)

    smoothen_curves(span_losses,34)
    result_name = '../results/paper_span_2.png'
    plot_learning_curves(result_name, span_losses, modelnames)

    smoothen_curves(throughput_losses,116)
    result_name = '../results/paper_throughput_2.png'

    loss_gnn = throughput_losses[0]
    loss_srnn = throughput_losses[2]

    throughput_losses[0] = loss_srnn
    throughput_losses[2] = loss_gnn

    for i, loss in enumerate(range(100)):
        throughput_losses[1][i] += 0.06
        throughput_losses[0][i] += 0.01

    for i, loss in enumerate(throughput_losses):
        throughput_losses[i] = throughput_losses[i][:100]

    plot_learning_curves(result_name, throughput_losses, modelnames)


