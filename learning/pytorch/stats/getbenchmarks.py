from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import common_libs.utilities as ut
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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


def plot_histogram(filename, values):
    plt.figure()
    plt.hist(values, bins=50, range=(0,1000), edgecolor='black', linewidth=0.3)
    plt.xlabel('throughput for 100 repetitions (cycles)')
    plt.ylabel('count')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


class Program:

    def __init__(self, name):
        self.name = name
        self.basicblocks = 0
        self.times = dict()

class Benchmark:

    def __init__(self, name):
        self.name = name
        self.programs = []

    def add_program(self, name):
        program = Program(name)
        self.programs.append(program)

    def add_programs(self, names):
        for name in names:
            self.add_program(name)




if __name__ == '__main__':

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',action='store', type=str, required=True)
    parser.add_argument('--config',action='store', type=str, required=True)
    parser.add_argument('--arch',action='store', type=int, required=True)
    args = parser.parse_args(sys.argv[1:])

    #setting up
    sym_dict,_ = ut.get_sym_dict()
    offsets = ut.read_offsets()

    opcode_start = offsets[0]
    operand_start = offsets[1]
    int_immed = offsets[2]
    float_immed = offsets[3]
    mem_start = offsets[4]

    costs = dict()
    for i in range(opcode_start, mem_start):
        costs[i] = 1

    linux = Benchmark('linux')
    spec2006 = Benchmark('SPEC2006')
    spec2017 = Benchmark('SPEC2017')
    nas = Benchmark('NAS')
    polybench = Benchmark('polybench-3.1')
    tsvc = Benchmark('TSVC')
    cortexsuite = Benchmark('cortexsuite')
    simd = Benchmark('simd')


    allbench = [linux, spec2006, spec2017, nas, polybench, tsvc, cortexsuite, simd]

    cnx = ut.create_connection_from_config(args.config,args.database)

    sql = 'select distinct program from code'
    rows = ut.execute_query(cnx, sql, True)

    print 'distinct programs found : ' + str(len(rows))

    tsvc.add_program('runvec')
    tsvc.add_program('runnovec')
    linux.add_programs(['ld-linux-x86-64.so.2', 'linux-vdso.so.1'])
    polybench.add_programs(['2mm','3mm','atax','bicg','cholesky','doitgen','gemm','gemver','gesummv','mvt','symm','syrk','syr2k','trisolv', 'trmm', 'durbin','dynprog','gramschmidt','lu','ludcmp','correlation','covariance','floyd-warshall','reg_detect','adi','fdtd-2d','fdtd-apml','jacobi-1d-imper','jacobi-2d-imper','seidel-2d'])
    linux.add_program('echo')
    cortexsuite.add_programs(['sphinx','a.out'])
    spec2006.add_program('specxz')


    for row in rows:

        if '_r_' in row[0] or '_s_' in row[0] or 'mytest-m64' in row[0]:
            spec2017.add_program(row[0])
        elif '.A' in row[0]:
            nas.add_program(row[0])
        elif 'test_' in row[0]:
            simd.add_program(row[0])
        elif '_base.gcc' in row[0]:
            spec2006.add_program(row[0])
        elif 'lib' in row[0]:
            linux.add_program(row[0])
        elif '-small' in row[0]:
            cortexsuite.add_program(row[0])

        found = False
        for bench in allbench:
            for program in bench.programs:
                if program.name == row[0]:
                    found = True
                    break
        if not found:
            print 'not in any benchmark : ' + str(row[0])


    total = 0
    for bench in allbench:
        total += len(bench.programs)

    print 'all programs (pre-populated included) : ' + str(total)


    #ok now for the statistics
    for bench in allbench:
        for program in bench.programs:

            sql = 'select count(*) from code where program=\'' + program.name + '\''
            #print sql
            row = ut.execute_query(cnx, sql, True)

            assert len(row) == 1

            program.basicblocks = int(row[0][0])


    #times available
    sql = 'select code_id, program from code'
    rows = ut.execute_query(cnx, sql, True)


    time_available = dict()
    timevals = dict()
    timevals_test = dict()

    for i in range(1, args.arch + 1):
        time_available[i] = 0
        timevals[i] = []
        timevals_test[i] = []

    start_test = int(len(rows) * 0.8)
    idx = 0

    for row in tqdm(rows):

        code_id = row[0]
        idx += 1

        for i in range(1, args.arch + 1):

            sql = 'select time from times where code_id=' + str(code_id) + ' and kind=\'actual\' and arch=' + str(i)
            times = ut.execute_query(cnx, sql, True)

            if len(times) > 0:
                time_available[i] += 1
                timevals[i].append(times[0][0])
                if idx > start_test:
                    timevals_test[i].append(times[0][0])
                for bench in allbench:
                    for program in bench.programs:
                        if program.name == row[1]:
                            if i in program.times:
                                program.times[i] += 1
                            else:
                                program.times[i] = 1


    print 'available times for various architectures'
    print time_available

    ithemal_home = os.environ['ITHEMAL_HOME'] + '/learning/pytorch'

    for i in timevals.keys():
        plot_histogram(ithemal_home + '/results/figures/timinghist_' + str(i) + '.png', timevals[i])
        plot_histogram(ithemal_home + '/results/figures/timinghist_test_' + str(i) + '.png', timevals_test[i])


    #print it out

    print '\nstatistics for each benchmark'
    print 'name -- bbs -- timed bbs'

    for i in range(1, args.arch + 1):

        print '\narch : ' + str(i) + '\n'

        tbbs = 0
        tprograms = 0
        ttimes = 0

        for bench in allbench:

            bbs = 0
            programs = 0
            times = 0

            for program in bench.programs:
                programs += 1
                bbs += program.basicblocks
                if i in program.times:
                    times += program.times[i]

            print bench.name + ' ' + str(programs) + ' ' + str(bbs) + ' ' + str(times)

            tbbs += bbs
            tprograms += programs
            ttimes += times

        print 'Total ' + str(tprograms) + ' ' + str(tbbs) + ' ' + str(ttimes)



    cnx.close()
