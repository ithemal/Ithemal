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
        plt.savefig(self.name + '.png')
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

if __name__ == '__main__':

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
    maxnum = 20
    for i in range(opcode_start, mem_start):
        costs[i] = 1

    cnx = ut.create_connection('static')

    sql = 'SELECT code_token, code_text, time from code'
    rows = ut.execute_query(cnx, sql, True)

    stats = CodeStats()
    stats.insert_field('ins')
    stats.insert_field('span')
    stats.insert_field('opcodes')
    stats.insert_field('time')
    
    incorrect_time = 0
    correct_time = 0

    x_instrs = []
    y_times = []

    for row in rows:
        
        if row[1] != '' and row[2] != None:
            
            tokens_text = row[0].split(',')
            tokens = []
            for i in tokens_text:
                if i != '':
                    tokens.append(int(i))
            block = ut.create_basicblock(tokens)
            num_instrs = block.num_instrs()
            num_span = block.num_span(costs)

            for instr in block.instrs:
                stats.insert_value('opcodes', instr.opcode)
            
            stats.insert_value('ins', num_instrs)
            stats.insert_value('span', num_span)
            
            if row[2] <= 20 or row[2] > 100000:
                incorrect_time += 1
                #print row[1]
                #print row[2]
            else:
                correct_time += 1
                stats.insert_value('time', row[2])
                
                if row[2] <= 1000:
                    y_times.append(row[2])
                    x_instrs.append(num_instrs)
                if row[2] < 50 and num_instrs > 5:
                    print row[2]
                    print row[1]
                
    print incorrect_time, correct_time

    stats.plot_stats()
        
    plt.scatter(x_instrs, y_times)
    plt.savefig('timevsinstrs.png')
    plt.close()


    #opcode popularity printing
    opcode_stats = stats.get_stat('opcodes')
    opcode_dict = opcode_stats.value_dict

    total_ins = len(opcode_stats.values)
    print total_ins
    print total_ins / float(len(rows))

    sorted_keys = sorted(opcode_dict, key = opcode_dict.get, reverse=True)
    
    x = []
    y = []
    

    for key in sorted_keys:
        x.append(sym_dict[key])
        y.append(opcode_dict[key] / float(total_ins))


    x_pos = np.arange(len(x))

    maxnum = 30
    plt.bar(x_pos[:maxnum], y[:maxnum], align='center', alpha=0.5)
    plt.xticks(x_pos[:maxnum], x[:maxnum], rotation=80)
    plt.ylabel('percentage')
    plt.title('Opcode popularity')
    plt.savefig('opcode_pop.png', bbox_inches='tight')
    plt.close()


    cnx.close()
