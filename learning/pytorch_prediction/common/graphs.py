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

def plot_line_graphs(filename, losses, legend, ylabel='loss', xlabel='batch', title='Learning Curves'):
    plt.figure()
    for loss, label in zip(losses, legend):
        y = loss
        x = np.arange(len(loss))
        h = plt.plot(x,y, '.-', linewidth=1, markersize=2, label=label)
 
    plt.legend()
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)
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

    plot_line_graphs('test.png',ys,labels)


        

    
    
