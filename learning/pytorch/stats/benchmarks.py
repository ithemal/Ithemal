from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

import common_libs.utilities as ut
import models.graph_models as md
import models.train as tr
import data.data_cost as dt

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
        
    def add_bb(self, name):

        for program in self.programs:
            if name in program.name:
                program.basicblocks += 1
                return
                
        program = Program(name)
        self.programs.append(program)
        program.basicblocks = 1

    def add_programs(self, names):
        for name in names:
            self.add_program(name)


def print_benchmark(benchmark, name):

    tbbs = 0
    tprograms = 0
    
    for bench in benchmark:
        
        bbs = 0
        programs = 0
        
        # print 'b ' + bench.name
        for program in bench.programs:
            if program.basicblocks > 0:
                # print 'p ' +  program.name
                programs += 1
                bbs += program.basicblocks
                                
        tbbs += bbs
        tprograms += programs
          
    print name + ' ' + str(tprograms) + ' ' + str(tbbs)

        

if __name__ == '__main__':

    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',action='store', type=str, required=True)
    parser.add_argument('--arch',action='store', type=int, required=True)
    args = parser.parse_args(sys.argv[1:])

    #setting up
    # sym_dict,_ = ut.get_sym_dict()
    # offsets = ut.read_offsets()

    # opcode_start = offsets[0]
    # operand_start = offsets[1]
    # int_immed = offsets[2]
    # float_immed = offsets[3]
    # mem_start = offsets[4]

    # costs = dict()
    # for i in range(opcode_start, mem_start):
    #     costs[i] = 1

    linux = Benchmark('linux')
    spec2006 = Benchmark('SPEC2006')
    spec2017 = Benchmark('SPEC2017')
    nas = Benchmark('NAS')
    polybench = Benchmark('polybench-3.1')
    tsvc = Benchmark('TSVC')
    cortexsuite = Benchmark('cortexsuite')
    simd = Benchmark('simd')
    clang = Benchmark('clang')
    gimp = Benchmark('gimp')
    python = Benchmark('python')
    firefox = Benchmark('firefox')
    openoffice = Benchmark('open-office')
    sqlite = Benchmark('sqlite')
    games = Benchmark('games')
    java = Benchmark('java')
    rhythm = Benchmark('rhythm')

    allbench = [clang, spec2006, spec2017, cortexsuite, nas, polybench, tsvc, simd, gimp, python, firefox, openoffice, linux, sqlite, games, java, rhythm]


    with open(os.path.join(os.environ['ITHEMAL_HOME'], 'gimp.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            gimp.add_programs(line.split())
    with open(os.path.join(os.environ['ITHEMAL_HOME'], 'firefox.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            firefox.add_programs(line.split())
    with open(os.path.join(os.environ['ITHEMAL_HOME'], 'linux.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            linux.add_programs(line.split())

            
    cnx = ut.create_connection(args.database)

    tsvc.add_programs(['runvec'])
    tsvc.add_programs(['runnovec'])
    polybench.add_programs(['2mm','3mm','atax','bicg','cholesky','doitgen','gemm','gemver','gesummv','mvt','symm','syrk','syr2k','trisolv', 'trmm', 'durbin','dynprog','gramschmidt','lu','ludcmp','correlation','covariance','floyd-warshall','reg_detect','adi','fdtd-2d','fdtd-apml','jacobi-1d-imper','jacobi-2d-imper','seidel-2d'])
    cortexsuite.add_programs(['sphinx','a.out','liblinear-tsmall'])
    spec2006.add_programs(['specxz'])
    clang.add_programs(['clang'])
    java.add_programs(['javaldx','libjvmfwklo.so'])
    openoffice.add_programs(['oosplash','soffice.bin','libuno_cppu.so.3','libuno_sal.so.3','libuno_salhelpergcc3.so.3','libvclplug_genlo.so','libmergedlo.so'])
    linux.add_programs(['echo','gedit','libm.so.6','libc.so.6','libgcc_s.so.1','ld-linux-x86-64.so.2','libpthread.so.0','linux-vdso.so.1','libdl.so.2','libglib-2.0.so.0','libselinux.so.1','libdbus-1.so.3','libclucene-shared.so.1','libz.so.1','libcom_err.so.2','libexpat.so.1','libudev.so.1','libtinfo.so.5','libclucene-core.so.1','libpng12.so.0','libcrypto.so.1.0.0','_glib.so','_gobject.so','_hashlib.x86_64-linux-gnu.so','libgpg-error.so.0'])
    games.add_programs(['sgt-mines'])
    gimp.add_programs([])
    rhythm.add_programs(['rhythmbox','libmpris.so','libaudioscrobbler.so','libmmkeys.so'])
    
    # gimp.add_programs(['sparkle','noise_spread','contrast-stretch','engrave','threshold-alpha','gradient-flare','crop-auto','shift','jigsaw','waves','curve-bend','tile-glass','blur-motion','noise-hsv','lens-flare','semi-flatten','blur-gauss-selective','animation-optimize','colors-exchange','lens-apply','nova','cartoon','border-average','sharpen','channel-mixer','pixelize','red-eye-removal','color-to-alpha','edge-laplace','lens-distortion','nl-filter','warp','blinds','edge-sobel','file-png','max-rgb','colorify','convolution-matrix','emboss','colormap-remap','noise-solid','noise-rgb','polar-coords','iwarp','tile','grid','despeckle','rotate','sample-colorize','tile-seamless','blur-gauss','noise-randomize','blur','noise-spread','map-object','contrast-normalize','noise-slur.so','gradient-map','smooth-palette','contrast-retinex','whirl-pinch','color-enhance','illusion','edge-neon','crop-zealous','lighting','mosaic','apply-canvas','edge','edge-dog','color-cube-analyze','deinterlace','wind','antialias','ripple'])
    # linux.add_programs(['gedit'])
    # linux.add_programs(['ld-linux-x86-64.so.2', 'linux-vdso.so.1'])
  

    unaffiliated_bbs = 0
    unaffiliated_modules = set()

    sql = 'SELECT distinct(module) FROM code_metadata WHERE code_id IN (SELECT DISTINCT(code_id) FROM time WHERE kind_id=1 AND arch_id=' + str(args.arch) + ')'
    rows = ut.execute_query(cnx, sql, True)

    print 'distinct basic blocks found : ' + str(len(rows))

    
    
    for row in tqdm(rows):

        value = str(row[0])
        # check for prepopulated
        found = False
        bench_f = None
        program_f = None
        
        for bench in allbench:
            for program in bench.programs:
                if program.name.startswith(value) and (not (value == 'lu' and bench.name == 'NAS')):
                    if found:
                        print 'ori ' + bench_f
                        print 'ori ' + program_f
                        print value
                        print program.name
                        print bench.name
                    # assert(not found)
                    found = True
                    program_f = program.name
                    bench_f = bench.name
                    bench.add_bb(program.name)
                    break
            if found:
                break
                
        if not found:
            found = True
            if '_r_' in value or '_s_' in value or 'mytest-m64' in value:
                spec2017.add_bb(value)
            elif '.A' in value:
                nas.add_bb(value)
            elif 'gimp' in value:
                gimp.add_bb(value)
            elif 'test_' in value:
                simd.add_bb(value)
            elif '_base.gcc' in value:
                spec2006.add_bb(value)
            elif '-small' in value:
                cortexsuite.add_bb(value)
            elif 'clang' in value:
                clang.add_bb(value)
            elif 'python' in value:
                python.add_bb(value)
            elif 'firefox' in value:
                firefox.add_bb(value)
            elif 'sqlite' in value:
                sqlite.add_bb(value)
            #elif 'lib' in value:
            #    linux.add_bb(value)
            else:
                found = False
        

        if not found:
            unaffiliated_modules.add(value)
            unaffiliated_bbs += 1
            print(value)
        

    total = 0
    for bench in allbench:
        total += len(bench.programs)

    print 'all programs (pre-populated included) : ' + str(total)
    print 'unaffiliated bbs ' + str(unaffiliated_bbs)
    print 'total bbs ' + str(len(rows))

    print 'unaffiliated\n'
    for mod in unaffiliated_modules:
        print mod
    
    print '\nstatistics for each benchmark'
    print 'name -- bbs -- timed bbs'

    print '\narch : ' + str(args.arch) + '\n'

    tbbs = 0
    tprograms = 0

    compilers = [clang, python]
    consumer = [firefox, games, java, openoffice, rhythm, sqlite]

    for bench in allbench:

        bbs = 0
        programs = 0

        # print 'b ' + bench.name
        for program in bench.programs:
            if program.basicblocks > 0:
                # print 'p ' +  program.name
                programs += 1
                bbs += program.basicblocks

        print bench.name + ' ' + str(programs) + ' ' + str(bbs)

        tbbs += bbs
        tprograms += programs
          
    print 'Total ' + str(tprograms) + ' ' + str(tbbs)

    print 'Total from count ' + str(tbbs + unaffiliated_bbs)

    print_benchmark(compilers, 'compilers')
    print_benchmark(consumer, 'consumer')
        
    
    cnx.close()
