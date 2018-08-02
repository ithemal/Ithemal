from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

import common.utilities as ut
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


#classes for keeping statistics - cumulative (struct of arrays format)

class Stat:

    def __init__(self, name):
        self.name = name
        self.values = list()
        self.value_dict = dict()

    def insert(self, value):
        self.values.append(value)
        if value in self.value_dict:
            self.value_dict[value] += 1
        else:
            self.value_dict[value] = 1

    def print_values(self):
        print self.value_dict

    def plot_values(self, max_v):
        plt.figure()
        plt.hist(self.values, bins=max_v, range=(0,max_v), edgecolor='black', linewidth=1.2)
        plt.xlabel(self.name)
        plt.ylabel('count')
        plt.title(self.name + ' count histogram')
        plt.savefig('figures/' + self.name + '.png')
        plt.close()

        
class CodeStats:
    
    def __init__(self):
        self.stats = []

    def insert_field(self, field):
        self.stats.append(Stat(field))

    def insert_value(self, field, value):
        for stat in self.stats:
            if stat.name == field:
                stat.insert(value)

    def plot_stats(self):
        for stat in self.stats:
            stat.plot_values(1000)

    def print_stats(self):
        for stat in self.stats:
            stat.print_values()

    def get_stat(self, name):
        for stat in self.stats:
            if stat.name == name:
                return stat


#plotting and statistics extraction functions

def plot_tsne(embeddings, first_n, sym_dict, id2word, mem_offset, filename):

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(embeddings[1:first_n, :])
    labels = [ut.get_name(id2word[i],sym_dict,mem_offset) for i in xrange(1,first_n)]
        
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        font = matplotlib.font_manager.FontProperties(size=12)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     fontproperties=font,
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
                

def plot_3d_map(x,y,bins,xlabel,filename,xmax=None,ymax=None):

    #ins count vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(x), np.array(y), bins=bins)

    if xmax == None:
        xmax = xedges[-1]
    if ymax == None:
        ymax = yedges[-1]

    extent = [xedges[0], xmax, yedges[0], ymax]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    y_str = 'measured throughput (cycles)'
    cmap = plt.get_cmap('viridis_r')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(xmax/ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_str)
    plt.savefig(filename)
    plt.close()



def get_basic_statistics(rows, sym_dict, costs):

    #basic statistics printing
    stats = CodeStats()
    stats.insert_field('ins')
    stats.insert_field('span')
    stats.insert_field('opcodes')

    total_ins = 0

    for row in tqdm(rows):
        
        if row[1] != '' and row[2] != None:
            
            block = ut.create_basicblock(row[0])
            num_instrs = block.num_instrs()
            num_span = block.num_span(costs)
            
            total_ins += num_instrs

            for instr in block.instrs:
                stats.insert_value('opcodes', instr.opcode)
            
            stats.insert_value('ins', num_instrs)
            stats.insert_value('span', num_span)
            
    stats.get_stat('ins').plot_values(30)
    stats.get_stat('span').plot_values(30)
    stats.get_stat('opcodes').plot_values(1300)

    #opcode popularity printing
    opcode_stats = stats.get_stat('opcodes')
    opcode_dict = opcode_stats.value_dict

    total_ins = len(opcode_stats.values)
    print 'total instructions : ' + str(total_ins)
    print 'basic block density : ' + str(total_ins / float(len(rows)))

    sorted_keys = sorted(opcode_dict, key = opcode_dict.get, reverse=True)
    
    x = []
    y = []
    
    for key in sorted_keys:
        x.append(sym_dict[key])
        y.append(opcode_dict[key] / float(total_ins))


    x_pos = np.arange(len(x))

    maxnum = 30
    plt.figure()
    plt.bar(x_pos[:maxnum], y[:maxnum], align='center', alpha=0.5)
    plt.xticks(x_pos[:maxnum], x[:maxnum], rotation=80)
    plt.ylabel('percentage')
    plt.title('Opcode popularity')
    plt.savefig('figures/opcode_pop.png', bbox_inches='tight')
    plt.close()



def get_embedding_visualization(model_file, embed_file):

    data = dt.DataInstructionEmbedding()
    data.set_embedding(embed_file)
    data.read_meta_data()

    num_classes = 1
    embedding_size = data.final_embeddings.shape[1]
    model = md.GraphNN(embedding_size = embedding_size, hidden_size = 256, num_classes = num_classes)
    model.set_learnable_embedding(mode = 'none', dictsize = max(data.word2id) + 1, seed = data.final_embeddings)


    train = tr.Train(model, data)
    train.load_checkpoint(model_file)
    embeddings = model.final_embeddings.weight.detach().numpy()
    plot_tsne(embeddings, 200, data.sym_dict, data.id2word, data.mem_start, 'figures/tsne_none.png')

    embeddings = data.final_embeddings
    plot_tsne(embeddings, 200, data.sym_dict, data.id2word, data.mem_start, 'figures/tsne_word2vec.png')

    

def get_timing_statistics_accuracy(cnx,rows,save=None,load=None):

    ins_times = []
    actual_times = []
    additive_times = []
    predicted_times = []
    llvm_times = []
    iaca_times = []

    if load != None:
        f = open(load,'r')
        (ins_times, actual_times, additive_times, predicted_times) = pickle.load(f)
        f.close()
    else:

        for row in tqdm(rows):
        
            if row[2] != '' :
            
                sql = 'select kind, time from times where code_id=' + str(row[1])
                times = ut.execute_query(cnx, sql, True)

                #should we skip this?            
                actual = []
                predicted = []
                additive = []
                llvm = []
                iaca = []
            
                for time in times:
                    if time[0] == 'actual':
                        if time[1] < 1000 and time[1] > 20:
                            actual.append(time[1])
                    elif time[0] == 'add':
                        additive.append(time[1])
                    elif time[0] == 'predicted':
                        predicted.append(time[1])
                    elif time[0] == 'llvm':
                        llvm.append(time[1])
                    elif time[0] == 'iaca':
                        iaca.append(time[1])
                    

                if len(actual) == 0 or len(additive) == 0 or len(predicted) == 0 or len(llvm) == 0 or len(iaca) == 0:
                    continue

                block = ut.create_basicblock(row[0])
                num_instrs = block.num_instrs()

                ins_times.append(int(num_instrs))
                actual_times.append(int(actual[0]))
                additive_times.append(int(additive[0]))
                predicted_times.append(int(predicted[0]))
                llvm_times.append(int(llvm[0]))
                iaca_times.append(int(iaca[0]))
                
        
        if save != None:
            f = open(save,'w+')
            pickle.dump((ins_times, actual_times, additive_times, predicted_times), f)
            f.close()

    print 'no of times selected ' + str(len(ins_times))

    #correlation
    print 'correlation with ins count:'
    print np.corrcoef(ins_times, actual_times)
    print 'correlation with additive model:'
    print np.corrcoef(additive_times, actual_times)
    print 'correlation with nn model:'
    print np.corrcoef(predicted_times, actual_times)
    print 'correlation with llvm:'
    print np.corrcoef(llvm_times, actual_times)
    print 'correlation with iaca:'
    print np.corrcoef(iaca_times, actual_times)


    start = math.floor(len(ins_times) * 0.8)
    start = int(start)
    
    ins_times_test = ins_times[start:]
    additive_times_test = additive_times[start:]
    predicted_times_test = predicted_times[start:]
    llvm_times_test = llvm_times[start:]
    actual_times_test = actual_times[start:]
    iaca_times_test = iaca_times[start:]

    #test set
    print 'test set only'
    print 'correlation with ins count:'
    print np.corrcoef(ins_times_test, actual_times_test)
    print 'correlation with additive model:'
    print np.corrcoef(additive_times_test, actual_times_test)
    print 'correlation with nn model:'
    print np.corrcoef(predicted_times_test, actual_times_test)
    print 'correlation with llvm:'
    print np.corrcoef(llvm_times_test, actual_times_test)
    print 'correlation with iaca:'
    print np.corrcoef(iaca_times_test, actual_times_test)

    

    #ins count vs actual
    xmax = 40
    heatmap, xedges, yedges = np.histogram2d(np.array(ins_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xmax, yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    y_str = 'measured throughput (cycles)'
    cmap = plt.get_cmap('viridis_r')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    #ax.set_aspect(float(max(ins_times))/float(max(actual_times)))
    ax.set_aspect(xmax/float(max(actual_times)))
    ax.set_xlabel('instruction count')
    ax.set_ylabel(y_str)
    plt.savefig('figures/inscountheatmap.png')
    plt.close()


    #additive cost histogram
    plt.figure()
    plt.hist(additive_times, bins=50, range=(0,50), edgecolor='black', linewidth=0.3)
    plt.xlabel('additive times')
    plt.ylabel('count')
    plt.title('additive cost count histogram')
    plt.savefig('figures/additivehist.png')
    plt.close()


    #additive count vs actual
    xmax = 200
    heatmap, xedges, yedges = np.histogram2d(np.array(additive_times), np.array(actual_times), bins=(20,50))
    extent = [xedges[0], xmax, yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    #ax.set_aspect(float(max(additive_times))/float(max(actual_times)))
    ax.set_aspect(xmax/float(max(actual_times)))
    ax.set_xlabel('STOKE additive model')
    ax.set_ylabel(y_str)
    plt.savefig('figures/additiveheatmap.png')
    plt.close()


    #predicted vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(predicted_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(float(max(predicted_times))/float(max(actual_times)))
    ax.set_xlabel('data driven model')
    ax.set_ylabel(y_str)
    plt.savefig('figures/predictedheatmap.png')
    plt.close()

    #llvm vs actual
    xmax = 2000
    heatmap, xedges, yedges = np.histogram2d(np.array(llvm_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xmax, yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    #ax.set_aspect(float(max(llvm_times))/float(max(actual_times)))
    ax.set_aspect(xmax/float(max(actual_times)))
    ax.set_xlabel('llvm machine model')
    ax.set_ylabel(y_str)
    plt.savefig('figures/llvmheatmap.png')
    plt.close()



#this file is about extraction of data - not population of data or training

#training done in - experiments
#population done in - timing
    
if __name__ == '__main__':

    #command line arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode',action='store')
    args = parser.parse_args(sys.argv[1:])

    #setting up
    offsets_filename = '../inputs/offsets.txt'
    encoding_filename = '../inputs/encoding.h'

    sym_dict,_ = ut.get_sym_dict(offsets_filename, encoding_filename)
    offsets = ut.read_offsets(offsets_filename)
    
    print offsets
    opcode_start = offsets[0]
    operand_start = offsets[1]
    int_immed = offsets[2]
    float_immed = offsets[3]
    mem_start = offsets[4]

    costs = dict()
    for i in range(opcode_start, mem_start):
        costs[i] = 1

    ###### collecting evaluation results from here on - assuming everything's populated in the database#######

    cnx = ut.create_connection('costmodel')
    rows = ut.get_data(cnx, 'text', ['code_id','code_intel'])

    #visualizing learnt and word2vec embedding for the code sequences
    get_embedding_visualization('../saved/graphCostNone.mdl', '../inputs/code_delim.emb')

    #statistics about the basic blocks we have collected - dataset
    get_basic_statistics(rows, sym_dict, costs)

    #generates correlations, root mean square errors and heatmaps/3d height maps - accuracy claim
    get_timing_statistics_accuracy(cnx, rows, save='saved/timing.pkl')

    #generates correlations, root mean square errors and heatmaps/3d height maps - portability claim
    get_timing_statistics_portability()

    #get throughput of prediction - note that each method should support this mode
    get_throughput_of_prediction()
    
    cnx.close()
