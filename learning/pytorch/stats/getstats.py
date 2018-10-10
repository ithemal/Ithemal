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

class TimeDist:

    def __init__(self, times):

        self.times = []
        self.counts = []
        for time in times:
            added = False
            for i, comptime in enumerate(self.times):
                if ((float(comptime) - float(time)) / float(comptime)) < 0.1 :
                    count = self.counts[i]
                    self.times[i] = (comptime * count + time)/float(count + 1)
                    self.counts[i] += 1
                    added = True
                    break
            if not added:
                self.times.append(time)
                self.counts.append(1)

        #is there a dominant mode
        self.mode = -1
        self.count = -1
        self.mode_valid = True

        for count, time in zip(self.counts, self.times):

            if count > self.count:
                self.count = count
                self.mode = time
                self.mode_valid = True
            elif count == self.count:
                self.mode_valid = False


class Time:

    def __init__(self,code_id):
        self.times = dict()
        self.dist = dict()
        self.code_id = code_id

    def add_time(self, kind, time):
        if kind in self.times:
            self.times[kind].append(time)
        else:
            self.times[kind] = [time]

    def get_dist(self):
        for key in self.times.keys():
            times = self.times[key]
            self.dist[key] = TimeDist(times)


#############################################
# dataset generation functions
#############################################


#in array of struct format, a code_id can have many times
def get_times(cnx, rows, arch):

    times = []

    for row in tqdm(rows):

        if row[2] != '' :

            sql = 'select kind, time from times where code_id=' + str(row[1]) + ' and arch=' + str(arch)
            times_r = ut.execute_query(cnx, sql, True)

            time = Time(row[1])
            times.append(time)

            for time_r in times_r:
                if time_r[0] == 'actual':
                    if time_r[1] < 10000 and time_r[1] > 20:
                        time.add_time(time_r[0], time_r[1])
                else:
                    time.add_time(time_r[0], time_r[1])

            block = ut.create_basicblock(row[0])
            num_instrs = block.num_instrs()
            time.add_time('num_instr',num_instrs)

    for time in tqdm(times):
        time.get_dist()

    return times

#outputs filtered sets in struct of arrays format
#total set, test set
def get_filtered_time_sets(times, percentage, total_kinds, test_kinds):

    print 'before removing invalid modes : ' + str(len(times))

    #valid actual times
    valid_times = dict()
    total_valid_times = 0

    for i, time in enumerate(times):
        valid_times[i] = False
        if 'actual' in time.times:
            actual_dist = time.dist['actual']

            #if there are contradicting values throw them away - cleaning the dataset
            if actual_dist.mode_valid:
                assert actual_dist.mode > 20 and actual_dist.mode < 10000
                valid_times[i] = True
                total_valid_times += 1


    print 'total valid times : ' + str(total_valid_times)

    total_set = dict()
    test_set = dict()

    test_start = int(total_valid_times * percentage)
    time_idx = 0

    for kind in total_kinds:
        total_set[kind] = []

    for kind in test_kinds:
        test_set[kind] = []

    #ok now let's get the sets
    for i, time in enumerate(times):
        if valid_times[i]:

            #total set and plot total set
            is_total = True
            for kind in total_kinds:
                if kind not in time.times:
                    is_total = False
                    break

            if is_total:
                for kind in total_kinds:
                    if kind == 'actual': #actual get the mode
                        total_set[kind].append(time.dist[kind].mode)
                    else: #otherwise get the latest
                        total_set[kind].append(time.times[kind][-1])

            #test set and plot test set

            if time_idx >= test_start:

                is_test = True
                for kind in test_kinds:
                    if kind not in time.times:
                        is_test = False
                        break

                if is_test:
                    for kind in test_kinds:
                        if kind == 'actual':
                            test_set[kind].append(time.dist[kind].mode)
                        else:
                            test_set[kind].append(time.times[kind][-1])

            time_idx += 1

    return (total_set, test_set)


#filter times based on some criteria - this is for additional filtering
#struct of arrays to struct of arrays
def get_subset(times, filter_fn):

    kinds = times.keys()

    new_times = dict()
    for kind in kinds:
        new_times[kind] = []


    actual_times = times['actual']

    for i, time in enumerate(actual_times):

        if not filter_fn(time):
            for kind in kinds:
                new_times[kind].append(times[kind][i])

    return new_times

def rmse(xs,ys):

    error = []

    for x, y in zip(xs, ys):

        error.append(abs(float(x) - float(y)) / y)

    return statistics.mean(error)


def filter_timing_sets(times, percentage, filter_fn, total_kinds, test_kinds):

    (total_set, test_set) = get_filtered_time_sets(times, percentage, total_kinds, test_kinds)

    plot_total_set = get_subset(total_set, filter_fn)
    plot_test_set = get_subset(test_set, filter_fn)

    return (total_set, test_set, plot_total_set, plot_test_set)


def generate_timing_sets(cnx, rows, arch, kinds, percentage):

    #get haswell times

    print 'generating timing sets for arch ' + str(arch)

    times = get_times(cnx, rows, arch)

    time_filename = 'saved/times_' + str(arch) + '.pkl'
    with open(time_filename, 'w+') as f:
        pickle.dump(times, f)

    iaca_kinds = ['actual','iaca']

    all_kinds_times = filter_timing_sets(times, percentage, lambda time : time >= 1000, kinds, kinds)
    iaca_times = filter_timing_sets(times, 0.0, lambda time : time >= 1000, iaca_kinds, iaca_kinds)

    all_kinds_times_filename = 'saved/all_kinds_times_' + str(arch) + '.pkl'
    with open(all_kinds_times_filename, 'w+') as f:
        pickle.dump(all_kinds_times, f)

    iaca_times_filename = 'saved/iaca_times_' + str(arch) + '.pkl'
    with open(iaca_times_filename, 'w+') as f:
        pickle.dump(iaca_times, f)

def generate_static_statistics(rows, costs):

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

    with open('saved/static_stats.pkl','w+') as f:
        pickle.dump(stats, f)

#############################################
# plotting and statistic extraction functions
#############################################

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


def plot_3d_map(x,y,arch,bins,xlabel,filename,xmax=None,ymax=None):

    arch_strings = ['Haswell','Skylake','Nehalem']

    xs = []
    ys = []

    errors = 0
    for xt, yt in zip(x,y):
        if xt >= 1000 or yt >= 1000:
            errors += 1
            continue
        else:
            xs.append(xt)
            ys.append(yt)

    print errors

    #ins count vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(xs), np.array(ys), bins=bins)

    #print xedges
    #print yedges

    if xmax == None:
        xmax = xedges[-1]
    if ymax == None:
        ymax = yedges[-1]

    extent = [xedges[0], xmax, yedges[0], ymax]
    lognorm = matplotlib.colors.LogNorm(vmin = 20, vmax = heatmap.T.max(), clip = False)

    y_str = 'Predicted Throughput'
    x_str = 'Measured Throughput'
    cmap = plt.get_cmap('Reds')
    #cmap.set_clim(50, heatmap.T.max())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, cmap=cmap, norm=lognorm, extent=extent, origin='lower', clim=(20, heatmap.T.max()))
    cbar = plt.colorbar()
    plt.tick_params(labelsize=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set_aspect(xmax/ymax)
    ax.set_xlabel(x_str, fontsize=14)
    ax.set_ylabel(y_str, fontsize=14)
    ax.set_title(arch_strings[arch-1], fontsize=16)
    plt.savefig(filename)
    plt.close()


def plot_error_graphs(filename, times, iaca, kinds, labels, max, bins):

    interval = max / bins
    fig = plt.figure()
    ax = fig.add_subplot(111)


    for kind, label in zip(kinds, labels):


        if kind not in times:
            continue

        y = times['actual']
        x = times[kind]


        errors = [0] * bins
        count = [0] * bins
        errors_all = dict()

        for i in range(bins):
            errors_all[i] = [0]

        for x,y in zip(x, y):

            error = abs(x - y) / float(y)

            bin = int(math.floor(float(y) / interval))
            if bin >= bins:
                print error,y, bin


            errors[bin] = (errors[bin] * count[bin] + error) / float(count[bin] + 1)
            count[bin] += 1

            errors_all[bin].append(error)

        error_high = [0] * bins
        error_low = [0] * bins

        for i in range(bins):
            error_low[i] = errors[i] - np.percentile(errors_all[i],25)
            error_high[i] = np.percentile(errors_all[i],75) - errors[i]

        #plt.plot(range(0,max,interval),errors, '.-', linewidth=0.5, markersize=3, label=label)
        ax.errorbar(range(0,max,interval), errors, fmt='.-', linewidth=0.5, markersize=3, label=label, yerr=[error_low, error_high])

    x = iaca['iaca']
    y = iaca['actual']

    if len(x) > 0:
        errors = [0] * bins
        count = [0] * bins
        errors_all = dict()

        for i in range(bins):
            errors_all[i] = [0]


        for x,y in zip(x, y):

            error = abs(x - y) / float(y)

            bin = int(math.floor(float(y) / interval))

            errors[bin] = (errors[bin] * count[bin] + error) / float(count[bin] + 1)
            count[bin] += 1

            errors_all[bin].append(error)


        for i in range(bins):
            error_low[i] = errors[i] - np.percentile(errors_all[i],25)
            error_high[i] = np.percentile(errors_all[i],75) - errors[i]


        #plt.plot(range(0,max,interval),errors, '.-', linewidth=0.5, markersize=3, label='IACA')
        ax.errorbar(range(0,max,interval), errors, fmt='.-', linewidth=0.5, markersize=3, label='IACA', yerr=[error_low, error_high])

    plt.legend()
    plt.ylabel('Average Error')
    plt.xlabel('Throughput (cycles)')

    cur_xmin, cur_xmax = plt.xlim()
    cur_ymin, cur_ymax = plt.ylim()

    ymin = 0
    ymax = 1
    plt.ylim(ymin = ymin)
    plt.ylim(ymax = ymax)
    plt.savefig(filename)
    plt.close()


def compute_errors_cross_correlations(times, kinds, arch):

    (total_set, test_set, plot_total_set, plot_test_set) = times

    #print out statistics
    print 'total set ' + str(len(total_set['actual']))
    print 'total plot set ' + str(len(plot_total_set['actual']))
    print 'test set ' + str(len(test_set['actual']))
    print 'test plot set ' + str(len(plot_test_set['actual']))

    print '\ntotal time ' + str(arch)
    #total time correlation
    for kind in kinds:
        corr = np.corrcoef(total_set[kind], total_set['actual'])[0][1]
        print 'correlation with ' + kind + ' : ' + str(corr)
        e = rmse(total_set[kind], total_set['actual'])
        print 'rmse ' + kind + ' : ' + str(e)


    #test time correlation
    print '\ntest time ' + str(arch)
    for kind in kinds:
        corr = np.corrcoef(test_set[kind], test_set['actual'])[0][1]
        print 'correlation with ' + kind + ' : ' + str(corr)
        e = rmse(test_set[kind], test_set['actual'])
        print 'rmse ' + kind + ' : ' + str(e)

    #plot total time correlation
    print '\nplot total time ' + str(arch)
    for kind in kinds:
        corr = np.corrcoef(plot_total_set[kind], plot_total_set['actual'])[0][1]
        print 'correlation with ' + kind + ' : ' + str(corr)
        e = rmse(plot_total_set[kind], plot_total_set['actual'])
        print 'rmse ' + kind + ' : ' + str(e)


    #plot test time correlation
    print '\nplot test time ' + str(arch)
    for kind in kinds:
        corr = np.corrcoef(plot_test_set[kind], plot_test_set['actual'])[0][1]
        print 'correlation with ' + kind + ' : ' + str(corr)
        e = rmse(plot_test_set[kind], plot_test_set['actual'])
        print 'rmse ' + kind + ' : ' + str(e)


def plot_heatmaps_and_error_curves(times, iaca_times, arch):

    (total_set, test_set, plot_total_set, plot_test_set) = times
    (iaca_total_set,iaca_test_set,plot_iaca_total_set,plot_iaca_test_set) = iaca_times

    #get error graphs
    filename = 'figures/allsystems_errors_' + str(arch) + '.png'
    kinds = ['predicted','llvm']
    labels = ['Ithemal','llvm-mca']
    plot_error_graphs(filename, plot_test_set, plot_iaca_test_set, kinds, labels, 1000, 50)

    #total heatmaps
    #predicted vs actual
    plot_3d_map(plot_total_set['actual'],plot_total_set['predicted'],arch,50,'Data driven model','figures/predictedheatmap_' + str(arch) + '.png',xmax=1000,ymax=1000)
    #llvm vs actual
    plot_3d_map(plot_total_set['actual'],plot_total_set['llvm'],arch,50,'LLVM machine  model','figures/llvmheatmap_' + str(arch) + '.png',xmax=1000,ymax=1000)

    #test heatmaps
    #predicted vs actual
    plot_3d_map(plot_test_set['actual'],plot_test_set['predicted'],arch,50,'Data driven model','figures/predictedheatmap_' + str(arch) + '_test.png',xmax=1000,ymax=1000)
    #llvm vs actual
    plot_3d_map(plot_test_set['actual'],plot_test_set['llvm'],arch,50,'LLVM machine  model','figures/llvmheatmap_' + str(arch) + '_test.png',xmax=1000,ymax=1000)
    #iaca vs actual
    if len(plot_iaca_test_set['iaca']) > 0:
        plot_3d_map(plot_iaca_test_set['actual'],plot_iaca_test_set['iaca'],arch,50,'IACA  model','figures/iacaheatmap_' + str(arch) + '_test.png',xmax=1000,ymax=1000)


############################################################
# main entry functions for getting plots and collecting data
############################################################

def get_accuracy():


    all_kinds = ['predicted','llvm']
    iaca_kinds = ['iaca']

    for arch in range(1,4):

        with open('saved/all_kinds_times_' + str(arch) + '.pkl','r') as f:
            all_times = pickle.load(f)

        with open('saved/iaca_times_' + str(arch) + '.pkl','r') as f:
            iaca_times = pickle.load(f)

        compute_errors_cross_correlations(all_times,all_kinds,arch)
        if len(iaca_times[3]['iaca']) > 0:
            compute_errors_cross_correlations(iaca_times,iaca_kinds,arch)
        else:
            print 'iaca times not available for arch ' + str(arch)

        plot_heatmaps_and_error_curves(all_times,iaca_times,arch)


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


def get_basic_statistics(sym_dict):

    with open('saved/static_stats.pkl','r') as f:
        stats = pickle.load(f)

    instr_counts = stats.get_stat('ins')

    total_ins = 0
    for count in instr_counts.values:
        total_ins += count

    stats.get_stat('ins').plot_values(30)
    stats.get_stat('span').plot_values(30)
    stats.get_stat('opcodes').plot_values(1300)

    #opcode popularity printing
    opcode_stats = stats.get_stat('opcodes')
    opcode_dict = opcode_stats.value_dict

    total_ins = len(opcode_stats.values)
    print 'total instructions : ' + str(total_ins)
    print 'basic block density : ' + str(total_ins / float(len(instr_counts.values)))

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


def find_mispredicted_blocks(cnx, rows, save=None, load=None):

    times = get_times(cnx, rows, 1, save=save, load=load)

    kinds = ['actual','predicted','llvm','iaca']

    code_id_hashmap = dict()

    for row in rows:
        code_id_hashmap[row[1]] = row

    total = 0
    total_iaca_wrong = 0
    total_iaca_right = 0
    total_llvm_right = 0
    total_llvm_wrong = 0
    both = 0

    for time in times:

        complete = True
        for kind in kinds:
            if kind not in time.times:
                complete = False
                break

        if complete and time.dist['actual'].mode_valid:

            total += 1

            actual_time = time.dist['actual'].mode
            llvm_time = time.times['llvm'][-1]
            predicted_time = time.times['predicted'][-1]
            iaca_time = time.times['iaca'][-1]

            #where llvm is wrong
            if abs(actual_time - llvm_time) / actual_time > 0.2:
                print actual_time, llvm_time, iaca_time, predicted_time
                row = code_id_hashmap[time.code_id]
                print row[2]

            iaca_wrong = False
            llvm_wrong = False

            #where we are wrong but iaca is more right
            if (abs(actual_time - iaca_time) / float(actual_time)) > (abs(actual_time - predicted_time) / float(actual_time)):
               total_iaca_right += 1
               iaca_wrong = True
            else:
               total_iaca_wrong += 1

            #where llvm is more right or wrong
            if (abs(actual_time - llvm_time) / float(actual_time)) > (abs(actual_time - predicted_time) / float(actual_time)):
                total_llvm_right += 1
                llvm_wrong = True
            else:
                total_llvm_wrong += 1

            if iaca_wrong and llvm_wrong:

                row = code_id_hashmap[time.code_id]
                block = ut.create_basicblock(row[0])
                num_instrs = block.num_instrs()
                #if num_instrs > 5:
                    #print actual_time, llvm_time, iaca_time, predicted_time
                    #print row[2]


                both += 1



    print total
    print total_iaca_right
    print total_iaca_wrong
    print total_llvm_right
    print total_llvm_wrong
    print both


#this file is about extraction of data - not population of data or training

#training done in - experiments
#population done in - timing

if __name__ == '__main__':

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',action='store',type=int,default=2)
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

    if args.mode == 1:
        cnx = ut.create_connection('costmodel')
        rows = ut.get_data(cnx, 'text', ['code_id','code_intel'])

        #generate timing sets
        kinds = ['actual','predicted','llvm']
        n_kinds = ['actual','predicted','llvm']

        for arch in range(1,4):
            if arch != 3:
                generate_timing_sets(cnx, rows, arch, kinds, 0.8)
            else:
                generate_timing_sets(cnx, rows, arch, n_kinds, 0.8)

        #generate static stats
        generate_static_statistics(rows, costs)

        cnx.close()

    elif args.mode == 2:
        #statistics about the basic blocks we have collected - dataset
        #get_basic_statistics(sym_dict)

        #visualizing learnt and word2vec embedding for the code sequences
        #get_embedding_visualization('../saved/graph_none_haswell.mdl', '../inputs/code_delim.emb')

        #heatmaps and errors
        get_accuracy()

        #miscellanious - for other paper parts
        #find_mispredicted_blocks(cnx, rows[800000:])

