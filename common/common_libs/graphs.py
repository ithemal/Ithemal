import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import utilities as ut
import numpy as np
import random



def plot_histogram(filename, values, maxvalue, xlabel, ylabel, title):
    plt.figure()
    plt.hist(self.values, bins=maxvalue, range=(0,maxvalue), edgecolor='black', linewidth=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_line_graphs(filename, losses, legend, ylabel='loss', xlabel='batch', title='Learning Curves', xmin = None, xmax = None, ymin = None, ymax = None):
    plt.figure()
    for loss, label in zip(losses, legend):
        y = loss
        x = np.arange(len(loss))
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



if __name__ == '__main__':

    ys = []
    labels = ['graph1', 'graph2']

    for _ in range(2):
        y = []
        for i in range(random.randint(1,100)):
            y.append(random.randint(0,100))
        ys.append(y)

    plot_line_graphs('test.png',ys,labels, xmin=0, xmax=50, ymin=0, ymax=40)






