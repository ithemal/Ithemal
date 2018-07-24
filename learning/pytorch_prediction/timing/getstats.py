from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import common.utilities as ut
from tqdm import tqdm
import subprocess
import os
import re
import time
import argparse
import statistics
import pickle


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


def get_basic_statistics(rows, sym_dict, costs):

    #basic statistics printing
    stats = CodeStats()
    stats.insert_field('ins')
    stats.insert_field('span')
    stats.insert_field('opcodes')
    stats.insert_field('time')

    incorrect_time = 0
    correct_time = 0
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
            
            if row[2] <= 20 or row[2] > 100000:
                incorrect_time += 1
            else:
                correct_time += 1
                stats.insert_value('time', row[2])
    
    
    stats.get_stat('ins').plot_values(30)
    stats.get_stat('span').plot_values(30)
    stats.get_stat('opcodes').plot_values(1300)
    stats.get_stat('time').plot_values(1000)


    #opcode popularity printing
    opcode_stats = stats.get_stat('opcodes')
    opcode_dict = opcode_stats.value_dict

    total_ins = len(opcode_stats.values)
    print 'total instructions : ' + str(total_ins)
    print 'basic block density : ' + str(total_ins / float(len(rows)))
    print 'correct times : ' + str(correct_time)
    print 'incorrect times : ' + str(incorrect_time)

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



class Time:

    def __init__(self, name):
        self.name = name
        self.times = []
    
    def add_time(self, time):
        self.times.append(time)
                

def get_timing_related_statistics(cnx,rows,save=None,load=None):

    ins_times = []
    actual_times = []
    additive_times = []
    predicted_times = []
    llvm_times = []

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
                    

                if len(actual) == 0 or len(additive) == 0 or len(predicted) == 0 or len(llvm) == 0:
                    continue

                block = ut.create_basicblock(row[0])
                num_instrs = block.num_instrs()

                ins_times.append(int(num_instrs))
                actual_times.append(int(actual[0]))
                additive_times.append(int(additive[0]))
                predicted_times.append(int(predicted[0]))
                llvm_times.append(int(llvm[0]))
                
        
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

    #ins count vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(ins_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(float(max(ins_times))/float(max(actual_times)))
    ax.set_xlabel('ins count')
    ax.set_ylabel('timing for 100 reps')
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
    heatmap, xedges, yedges = np.histogram2d(np.array(additive_times), np.array(actual_times), bins=(20,50))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(float(max(additive_times))/float(max(actual_times)))
    ax.set_xlabel('additive model')
    ax.set_ylabel('timing for 100 reps')
    plt.savefig('figures/additiveheatmap.png')
    plt.close()


    #predicted vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(predicted_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(float(max(predicted_times))/float(max(actual_times)))
    ax.set_xlabel('data driven model')
    ax.set_ylabel('timing for 100 reps')
    plt.savefig('figures/predictedheatmap.png')
    plt.close()

    #llvm vs actual
    heatmap, xedges, yedges = np.histogram2d(np.array(llvm_times), np.array(actual_times), bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    lognorm = matplotlib.colors.LogNorm(vmin = 1, vmax = heatmap.T.max(), clip = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, norm=lognorm, extent=extent, origin='lower')
    plt.colorbar()
    ax.set_aspect(float(max(llvm_times))/float(max(actual_times)))
    ax.set_xlabel('llvm model')
    ax.set_ylabel('timing for 100 reps')
    plt.savefig('figures/llvmheatmap.png')
    plt.close()



    
if __name__ == '__main__':


    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',action='store')
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


    cnx = ut.create_connection('costmodel')
    rows = ut.get_data(cnx, 'text', ['code_id','code_intel'])

    #get_basic_statistics(rows, sym_dict, costs)
    get_timing_related_statistics(cnx,rows, save='saved/timing.pkl')

    cnx.close()
